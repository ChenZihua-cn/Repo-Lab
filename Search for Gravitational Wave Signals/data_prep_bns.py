"""
文件名: data_prep_bns.py
功能描述: 双中子星(BNS)引力波信号数据生成模块
    本模块用于生成双中子星旋近-并合-铃宕(Inspiral-Merger-Ringdown)过程的
    引力波时序数据，用于深度学习信号检测网络的训练和测试。

物理背景:
    双中子星(BNS)系统是引力波的重要波源。当两颗中子星相互绕转时，
    会辐射引力波并逐渐旋近，最终并合。与双黑洞(BBH)相比，BNS具有
    更长的旋近时间和更高的特征频率。

作者: Refactored from data_prep_bbh.py
日期: 2026

依赖库:
    - lal: LIGO算法库，提供物理常数和天文计算
    - lalsimulation: LIGO波形模拟库
    - numpy: 数值计算
    - scipy: 信号处理

使用示例:
    from data_prep_bns import sim_data, gen_par
    ts, par = sim_data(fs=8192, T_obs=1, snr=20, dets=['H1','L1'])

注意事项:
    - 需要安装LIGO的lalsuite库: pip install lalsuite
    - 中子星质量范围: 1.0-2.0 太阳质量
    - 总质量范围: 2.0-4.0 太阳质量
"""

from __future__ import division

# LIGO引力波数据分析库
import lal  # type: ignore
import lalsimulation  # type: ignore
from lal.antenna import AntennaResponse  # type: ignore
from lal import MSUN_SI, C_SI, G_SI  # type: ignore

import os
import sys
import argparse
import time
import numpy as np
import pickle as cPickle
from scipy.signal import filtfilt, butter
from scipy.optimize import brentq
from scipy import integrate, interpolate


# Python 2/3兼容性处理
if sys.version_info >= (3, 0):
    xrange = range

# 安全倍数因子：用于扩展时间窗口以避免边界效应
# 实际生成的数据时间长度 = safe * T_obs
safe = 2


class bnsparams:
    """
    双中子星物理参数类。
    
    存储双中子星系统的物理参数，包括质量、位置、方向等，
    用于引力波波形生成和信号分析。
    
    Attributes:
        mc (float): 啁啾质量(Chirp Mass)，单位：太阳质量
            Mc = (m1*m2)^(3/5) / (m1+m2)^(1/5)
            决定引力波频率演化的关键参数
        M (float): 总质量，单位：太阳质量
        eta (float): 对称质量比，η = m1*m2 / (m1+m2)^2，范围[0, 0.25]
        m1 (float): 主星质量（较重的中子星），单位：太阳质量
        m2 (float): 伴星质量（较轻的中子星），单位：太阳质量
        ra (float): 赤经(Right Ascension)，单位：弧度
        dec (float): 赤纬(Declination)，单位：弧度
        iota (float): 轨道倾角余弦值，cos(iota)
        phi (float): 参考相位，单位：弧度
        psi (float): 极化角，单位：弧度
        idx (int): 信号峰值在时序中的索引位置
        fmin (float): 信号起始频率，单位：Hz
        snr (float): 归一化后的单探测器信噪比
        SNR (float): 网络信噪比
    """
    
    def __init__(self, mc, M, eta, m1, m2, ra, dec, iota, phi, psi, idx, fmin, snr, SNR):
        """初始化BNS参数对象。"""
        self.mc = mc          # 啁啾质量
        self.M = M            # 总质量
        self.eta = eta        # 对称质量比
        self.m1 = m1          # 主星质量
        self.m2 = m2          # 伴星质量
        self.ra = ra          # 赤经
        self.dec = dec        # 赤纬
        self.iota = iota      # 轨道倾角余弦
        self.phi = phi        # 参考相位
        self.psi = psi        # 极化角
        self.idx = idx        # 峰值位置索引
        self.fmin = fmin      # 起始频率
        self.snr = snr        # 信噪比
        self.SNR = SNR        # 网络信噪比


def tukey(M, alpha=0.5):
    """
    生成Tukey窗函数（锥形余弦窗）。
    
    Tukey窗是矩形窗和余弦窗的组合，在时域两端有平滑的过渡，
    可以有效减少频谱泄漏，常用于引力波数据的加窗处理。
    
    数学定义:
        w[n] = 0.5 * (1 + cos(π*(2n/α/(M-1) - 1)))  for 0 <= n < α(M-1)/2
        w[n] = 1                                      for α(M-1)/2 <= n <= M-α(M-1)/2
        w[n] = 0.5 * (1 + cos(π*(2n/α/(M-1) - 2/α + 1))) for M-α(M-1)/2 < n < M
    
    Args:
        M (int): 窗函数长度（样本数）
        alpha (float): 锥形系数，范围[0, 1]
            alpha=0: 矩形窗
            alpha=1: 汉宁窗
    
    Returns:
        numpy.ndarray: Tukey窗函数数组，长度M
        
    Reference:
        代码改编自scipy.signal.windows.tukey
    """
    n = np.arange(0, M)
    width = int(np.floor(alpha * (M - 1) / 2.0))
    n1 = n[0:width + 1]
    n2 = n[width + 1:M - width - 1]
    n3 = n[M - width - 1:]

    # 左侧锥形过渡
    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (M - 1))))
    # 中间平坦部分
    w2 = np.ones(n2.shape)
    # 右侧锥形过渡
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (M - 1))))
    w = np.concatenate((w1, w2, w3))

    return np.array(w[:M])


def parser():
    """
    命令行参数解析器。
    
    定义数据生成脚本的所有命令行参数，包括采样率、观测时间、
    信噪比、探测器配置等。
    
    Returns:
        argparse.Namespace: 解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(
        prog='data_prep_bns.py',
        description='生成双中子星(BNS)引力波数据，用于深度学习网络训练。'
    )

    # 样本数量参数
    parser.add_argument('-N', '--Nsamp', type=int, default=7000,
                        help='总样本数量')
    parser.add_argument('-Nn', '--Nnoise', type=int, default=25,
                        help='每个信号的噪声实现数量（用于数据增强）')
    parser.add_argument('-Nb', '--Nblock', type=int, default=10000,
                        help='每个输出文件的训练样本数')
    
    # 信号参数
    parser.add_argument('-f', '--fsample', type=int, default=8192,
                        help='采样频率(Hz)')
    parser.add_argument('-T', '--Tobs', type=int, default=1,
                        help='观测持续时间(秒)')
    parser.add_argument('-s', '--snr', type=float, default=None,
                        help='信号积分信噪比(SNR)')
    parser.add_argument('-I', '--detectors', type=str, nargs='+',
                        default=['H1', 'L1'],
                        help='使用的探测器列表，如H1 L1')
    
    # 输出和配置
    parser.add_argument('-b', '--basename', type=str, default='test',
                        help='输出文件路径和基础名称')
    parser.add_argument('-m', '--mdist', type=str, default='astro',
                        help='质量分布类型(astro, gh, metric)')
    parser.add_argument('-z', '--seed', type=int, default=1,
                        help='随机数种子')

    return parser.parse_args()


def convert_beta(beta, fs, T_obs):
    """
    将beta值（定义中心输出窗口中期望时间段的比例）
    转换为完整安全时间窗口的索引。
    
    用于随机放置信号峰值在时间窗口中的位置。
    
    Args:
        beta (list): 包含两个元素的列表[beta_low, beta_high]，
            定义信号峰值可以放置的时间比例范围
        fs (int): 采样频率(Hz)
        T_obs (float): 观测时间(秒)
    
    Returns:
        tuple: (low_idx, high_idx) 信号峰值位置的索引范围
    """
    # 将beta转换到安全窗口坐标系
    newbeta = np.array([(beta[0] + 0.5 * safe - 0.5),
                        (beta[1] + 0.5 * safe - 0.5)]) / safe
    low_idx = int(T_obs * fs * newbeta[0])
    high_idx = int(T_obs * fs * newbeta[1])

    return low_idx, high_idx


def gen_noise(fs, T_obs, psd):
    """
    根据功率谱密度(PSD)生成有色噪声。
    
    使用频域方法生成符合给定PSD的随机噪声：
    1. 在频域生成白噪声
    2. 根据PSD的平方根进行缩放
    3. 逆傅里叶变换回时域
    
    Args:
        fs (int): 采样频率(Hz)
        T_obs (float): 观测时间(秒)
        psd (numpy.ndarray): 功率谱密度数组
    
    Returns:
        numpy.ndarray: 生成的噪声时序数据
        
    Note:
        噪声生成公式: n(t) = IFFT[sqrt(PSD/2) * (N_re + i*N_im)]
        其中N_re, N_im是标准正态分布随机数
    """
    N = T_obs * fs          # 总样本数
    Nf = N // 2 + 1         # 频域样本数
    dt = 1 / fs             # 采样时间间隔
    df = 1 / T_obs          # 频率分辨率

    # 频域振幅 = sqrt(PSD * T_obs / 4)
    amp = np.sqrt(0.25 * T_obs * psd)
    idx = np.argwhere(psd == 0.0)
    amp[idx] = 0.0
    
    # 生成复高斯随机数
    re = amp * np.random.normal(0, 1, Nf)
    im = amp * np.random.normal(0, 1, Nf)
    re[0] = 0.0  # DC分量为0
    im[0] = 0.0
    
    # 逆FFT得到时域噪声
    x = N * np.fft.irfft(re + 1j * im) * df

    return x


def gen_psd(fs, T_obs, op='AdvDesign', det='H1'):
    """
    生成各种探测器的功率谱密度(PSD)。
    
    根据探测器类型和运行阶段生成相应的噪声功率谱。
    目前支持LIGO Hanford(H1)和Livingston(L1)探测器。
    
    Args:
        fs (int): 采样频率(Hz)
        T_obs (float): 观测时间(秒)
        op (str): 探测器运行阶段选项：
            - 'AdvDesign': 设计灵敏度
            - 'AdvEarlyLow': 早期低灵敏度
            - 'AdvEarlyHigh': 早期高灵敏度
            - 'AdvMidLow': 中期低灵敏度
            - 'AdvMidHigh': 中期高灵敏度
            - 'AdvLateLow': 后期低灵敏度
            - 'AdvLateHigh': 后期高灵敏度
        det (str): 探测器名称('H1'或'L1')
    
    Returns:
        lal.REAL8FrequencySeries: LAL频率序列对象，包含PSD数据
    """
    N = T_obs * fs
    dt = 1 / fs
    df = 1 / T_obs
    psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0,
                                          df, lal.HertzUnit, N // 2 + 1)

    if det == 'H1' or det == 'L1':
        if op == 'AdvDesign':
            lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyLow':
            lalsimulation.SimNoisePSDAdVEarlyLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyHigh':
            lalsimulation.SimNoisePSDAdVEarlyHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidLow':
            lalsimulation.SimNoisePSDAdVMidLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidHigh':
            lalsimulation.SimNoisePSDAdVMidHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateLow':
            lalsimulation.SimNoisePSDAdVLateLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateHigh':
            lalsimulation.SimNoisePSDAdVLateHighSensitivityP1200087(psd, 10.0)
        else:
            print('未知噪声选项')
            exit(1)
    else:
        print('未知探测器')
        exit(1)

    return psd


def get_snr(data, T_obs, fs, psd, fmin):
    """
    计算信号的信噪比(SNR)。
    
    使用匹配滤波公式计算信号相对于探测器噪声的信噪比：
    SNR^2 = 4 * ∫|h(f)|^2 / S_n(f) df
    
    Args:
        data (numpy.ndarray): 时域信号数据
        T_obs (float): 观测时间(秒)
        fs (int): 采样频率(Hz)
        psd (numpy.ndarray): 功率谱密度
        fmin (float): 起始频率(Hz)
    
    Returns:
        float: 信号信噪比
    """
    N = T_obs * fs
    df = 1.0 / T_obs
    dt = 1.0 / fs
    fidx = int(fmin / df)

    # 应用Tukey窗减少频谱泄漏
    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd > 0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0 / psd[idx]

    # FFT变换到频域
    xf = np.fft.rfft(data * win) * dt
    
    # 计算SNR平方
    SNRsq = 4.0 * np.sum((np.abs(xf[fidx:]) ** 2) * invpsd[fidx:]) * df
    return np.sqrt(SNRsq)


def whiten_data(data, duration, sample_rate, psd, flag='td'):
    """
    根据PSD对数据进行白化处理。
    
    白化将有色噪声转换为白噪声，使得噪声在不同频率上具有相同的功率。
    这是引力波数据分析中的标准预处理步骤。
    
    数学公式: h_white(f) = h(f) / sqrt(S_n(f) / 2)
    
    Args:
        data (numpy.ndarray): 输入时序数据或频域数据
        duration (float): 数据持续时间(秒)
        sample_rate (int): 采样频率(Hz)
        psd (numpy.ndarray): 功率谱密度
        flag (str): 'td'表示时域输入，'fd'表示频域输入
    
    Returns:
        numpy.ndarray: 白化后的数据（时域或频域，取决于flag）
    """
    if flag == 'td':
        # 时域输入：先加窗再FFT
        win = tukey(duration * sample_rate, alpha=1.0 / 8.0)
        xf = np.fft.rfft(win * data)
    else:
        xf = data

    # 白化：除以PSD的平方根
    idx = np.argwhere(psd > 0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0 / psd[idx]
    xf *= np.sqrt(2.0 * invpsd / sample_rate)

    # DC分量设为0
    xf[0] = 0.0

    if flag == 'td':
        # 返回时域
        x = np.fft.irfft(xf)
        return x
    else:
        return xf


def gen_masses(m_min=1.0, M_max=4.0, mdist='astro', verbose=True):
    """
    从指定分布中抽取双中子星质量对。
    
    双中子星质量范围：
    - 单个中子星：1.0 - 2.0 M☉（太阳质量）
    - 总质量：2.0 - 4.0 M☉
    
    支持的质量分布：
    1. 'astro': 天体物理对数分布
    2. 'gh': George & Huerta分布
    3. 'metric': 度规基础分布
    
    Args:
        m_min (float): 最小质量，默认1.0 M☉
        M_max (float): 最大总质量，默认4.0 M☉
        mdist (str): 质量分布类型
        verbose (bool): 是否打印详细信息
    
    Returns:
        tuple: (m12, mc, eta)
            - m12 (numpy.ndarray): [m1, m2] 质量对
            - mc (float): 啁啾质量
            - eta (float): 对称质量比
    """
    flag = False
    
    if mdist == 'astro':
        if verbose:
            print('{}: 使用BNS天体物理对数质量分布'.format(time.asctime()))
        new_m_min = m_min
        new_M_max = M_max
        log_m_max = np.log(new_M_max - new_m_min)
        while not flag:
            # 对数均匀分布
            m12 = np.exp(np.log(new_m_min) + 
                        np.random.uniform(0, 1, 2) * (log_m_max - np.log(new_m_min)))
            flag = True if (np.sum(m12) < new_M_max) and \
                          (np.all(m12 > new_m_min)) and (m12[0] >= m12[1]) else False
        eta = m12[0] * m12[1] / (m12[0] + m12[1]) ** 2
        mc = np.sum(m12) * eta ** (3.0 / 5.0)
        return m12, mc, eta
        
    elif mdist == 'gh':
        if verbose:
            print('{}: 使用BNS的George & Huerta质量分布'.format(time.asctime()))
        m12 = np.zeros(2)
        while not flag:
            # BNS质量比通常接近1
            q = np.random.uniform(1.0, 2.0, 1)
            m12[1] = np.random.uniform(1.0, 2.0, 1)
            m12[0] = m12[1] * q
            flag = True if (np.all(m12 < 2.5)) and \
                          (np.all(m12 > 1.0)) and (m12[0] >= m12[1]) else False
        eta = m12[0] * m12[1] / (m12[0] + m12[1]) ** 2
        mc = np.sum(m12) * eta ** (3.0 / 5.0)
        return m12, mc, eta
        
    elif mdist == 'metric':
        if verbose:
            print('{}: 使用BNS度规基础质量分布'.format(time.asctime()))
        new_m_min = m_min
        new_M_max = M_max
        new_M_min = 2.0 * new_m_min
        eta_min = m_min * (new_M_max - new_m_min) / new_M_max ** 2
        while not flag:
            # 度规分布采样
            M = (new_M_min ** (-7.0 / 3.0) - 
                 np.random.uniform(0, 1, 1) * 
                 (new_M_min ** (-7.0 / 3.0) - new_M_max ** (-7.0 / 3.0))) ** (-3.0 / 7.0)
            eta = (eta_min ** (-2.0) - 
                   np.random.uniform(0, 1, 1) * 
                   (eta_min ** (-2.0) - 16.0)) ** (-1.0 / 2.0)
            m12 = np.zeros(2)
            m12[0] = 0.5 * M + M * np.sqrt(0.25 - eta)
            m12[1] = M - m12[0]
            flag = True if (np.sum(m12) < new_M_max) and \
                          (np.all(m12 > new_m_min)) and (m12[0] >= m12[1]) else False
        mc = np.sum(m12) * eta ** (3.0 / 5.0)
        return m12, mc, eta
    else:
        print('{}: 错误：未知质量分布'.format(time.asctime()))
        exit(1)


def get_fmin(M, eta, dt, verbose):
    """
    计算信号进入时间段的瞬时频率。
    
    使用2PN(后牛顿)阶啁啾时间公式反解频率。
    
    Args:
        M (float): 总质量（太阳质量）
        eta (float): 对称质量比
        dt (float): 到并合的时间(秒)
        verbose (bool): 是否打印信息
    
    Returns:
        float: 起始频率(Hz)
    """
    M_SI = M * MSUN_SI

    def dtchirp(f):
        """
        2PN阶啁啾时间公式。
        
        描述从频率f演化到并合(f→∞)所需的时间。
        """
        v = ((G_SI / C_SI ** 3) * M_SI * np.pi * f) ** (1.0 / 3.0)
        temp = (v ** (-8.0) + 
                ((743.0 / 252.0) + 11.0 * eta / 3.0) * v ** (-6.0) -
                (32 * np.pi / 5.0) * v ** (-5.0) + 
                ((3058673.0 / 508032.0) + 5429 * eta / 504.0 +
                 (617.0 / 72.0) * eta ** 2) * v ** (-4.0))
        return (5.0 / (256.0 * eta)) * (G_SI / C_SI ** 3) * M_SI * temp - dt

    # 在[1, 2000] Hz范围内求解
    fmin = brentq(dtchirp, 1.0, 2000.0, xtol=1e-6)
    if verbose:
        print('{}: 信号进入频段频率 = {} Hz'.format(time.asctime(), fmin))

    return fmin


def gen_par(fs, T_obs, mdist='astro', beta=[0.75, 0.95], verbose=True):
    """
    生成随机BNS参数集。
    
    随机生成双中子星系统的所有物理参数，用于波形模拟。
    
    Args:
        fs (int): 采样频率(Hz)
        T_obs (float): 观测时间(秒)
        mdist (str): 质量分布类型
        beta (list): 峰值位置比例范围[low, high]
        verbose (bool): 是否打印详细信息
    
    Returns:
        bnsparams: BNS参数对象
    """
    # BNS质量参数
    m_min = 1.0         # 单个中子星最小质量
    M_max = 4.0         # 双星系统最大总质量
    log_m_max = np.log(M_max - m_min)

    # 生成质量
    m12, mc, eta = gen_masses(m_min, M_max, mdist=mdist, verbose=verbose)
    M = np.sum(m12)
    if verbose:
        print('{}: BNS质量 = {},{} (啁啾质量 = {})'.format(
            time.asctime(), m12[0], m12[1], mc))

    # 生成轨道倾角（各向同性分布）
    iota = np.arccos(-1.0 + 2.0 * np.random.rand())
    if verbose:
        print('{}: BNS cos(倾角) = {}'.format(time.asctime(), np.cos(iota)))

    # 生成极化角 [0, 2π]
    psi = 2.0 * np.pi * np.random.rand()
    if verbose:
        print('{}: BNS 极化角 = {}'.format(time.asctime(), psi))

    # 生成参考相位 [0, 2π]
    phi = 2.0 * np.pi * np.random.rand()
    if verbose:
        print('{}: BNS 参考相位 = {}'.format(time.asctime(), phi))

    # 生成天空位置（球面上均匀分布）
    ra = 2.0 * np.pi * np.random.rand()
    dec = np.arcsin(-1.0 + 2.0 * np.random.rand())
    if verbose:
        print('{}: BNS 天空位置 = {},{}'.format(time.asctime(), ra, dec))

    # 随机选择峰值振幅位置
    low_idx, high_idx = convert_beta(beta, fs, T_obs)
    if low_idx == high_idx:
        idx = low_idx
    else:
        idx = int(np.random.randint(low_idx, high_idx, 1)[0])
    if verbose:
        print('{}: BNS 峰值时间 = {} s'.format(time.asctime(), idx / fs))

    # 中心区域起始索引
    sidx = int(0.5 * fs * T_obs * (safe - 1.0) / safe)

    # 计算起始频率
    fmin = get_fmin(M, eta, int(idx - sidx) / fs, verbose)
    if verbose:
        print('{}: 计算起始频率 = {} Hz'.format(time.asctime(), fmin))

    # 创建参数对象
    par = bnsparams(mc, M, eta, m12[0], m12[1], ra, dec,
                    np.cos(iota), phi, psi, idx, fmin, None, None)

    return par


def gen_bns(fs, T_obs, psds, snr=1.0, dets=['H1'], beta=[0.75, 0.95],
            par=None, verbose=True):
    """
    生成双中子星时域信号。
    
    使用LALSimulation生成BNS引力波波形，应用探测器响应，
    并归一化到指定信噪比。
    
    Args:
        fs (int): 采样频率(Hz)
        T_obs (float): 观测时间(秒)
        psds (list): 各探测器的PSD列表
        snr (float): 目标网络信噪比
        dets (list): 探测器名称列表
        beta (list): 峰值位置参数
        par (bnsparams): BNS参数对象
        verbose (bool): 是否打印信息
    
    Returns:
        tuple: (ts, hp, hc)
            - ts (numpy.ndarray): 探测器响应后的信号 (ndet, N)
            - hp (numpy.ndarray): 加极化信号
            - hc (numpy.ndarray): 叉极化信号
    """
    N = T_obs * fs
    dt = 1 / fs
    f_low = 15.0            # BNS起始频率（比BBH更高）
    amplitude_order = 0
    phase_order = 7
    approximant = lalsimulation.IMRPhenomD  # 使用IMRPhenomD近似
    dist = 1e6 * lal.PC_SI  # 1 Mpc参考距离

    # 生成波形，确保足够长
    flag = False
    while not flag:
        hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
            par.m1 * lal.MSUN_SI, par.m2 * lal.MSUN_SI,
            0, 0, 0, 0, 0, 0,
            dist,
            par.iota, par.phi, 0,
            0, 0,
            1 / fs,
            f_low, f_low,
            lal.CreateDict(),
            approximant)
        flag = True if hp.data.length > 2 * N else False
        f_low -= 1  # 逐步降低起始频率直到足够长
        
    orig_hp = hp.data.data
    orig_hc = hc.data.data

    # 计算参考索引（波形峰值）
    ref_idx = np.argmax(orig_hp ** 2 + orig_hc ** 2)

    # 中心区域起始索引
    sidx = int(0.5 * fs * T_obs * (safe - 1.0) / safe)

    # 创建Tukey窗（削减边界效应）
    win = np.zeros(N)
    tempwin = tukey(int((16.0 / 15.0) * N / safe), alpha=1.0 / 8.0)
    win[int((N - tempwin.size) / 2):
        int((N - tempwin.size) / 2) + tempwin.size] = tempwin

    # 遍历探测器
    ndet = len(psds)
    ts = np.zeros((ndet, N))
    hp = np.zeros((ndet, N))
    hc = np.zeros((ndet, N))
    intsnr = []
    j = 0
    
    for det, psd in zip(dets, psds):
        # 应用探测器响应和时间延迟
        ht_shift, hp_shift, hc_shift = make_bns(
            orig_hp, orig_hc, fs, par.ra, par.dec, par.psi, det, verbose)

        # 放置信号到时间序列
        ht_temp = ht_shift[int(ref_idx - par.idx):]
        hp_temp = hp_shift[int(ref_idx - par.idx):]
        hc_temp = hc_shift[int(ref_idx - par.idx):]
        
        if len(ht_temp) < N:
            ts[j, :len(ht_temp)] = ht_temp
            hp[j, :len(ht_temp)] = hp_temp
            hc[j, :len(ht_temp)] = hc_temp
        else:
            ts[j, :] = ht_temp[:N]
            hp[j, :] = hp_temp[:N]
            hc[j, :] = hc_temp[:N]

        # 应用窗函数
        ts[j, :] *= win
        hp[j, :] *= win
        hc[j, :] *= win

        # 计算信噪比
        intsnr.append(get_snr(ts[j, :], T_obs, fs, psd.data.data, par.fmin))

    # 归一化到目标信噪比
    intsnr = np.array(intsnr)
    scale = snr / np.sqrt(np.sum(intsnr ** 2))
    ts *= scale
    hp *= scale
    hc *= scale
    intsnr *= scale
    
    if verbose:
        print('{}: 网络信噪比 = {}'.format(time.asctime(), snr))

    return ts, hp, hc


def make_bns(hp, hc, fs, ra, dec, psi, det, verbose):
    """
    将h+和hx转换为探测器输出。
    
    应用探测器天线响应函数，并计算相对于地心的时间延迟。
    
    Args:
        hp (numpy.ndarray): 加极化波形
        hc (numpy.ndarray): 叉极化波形
        fs (int): 采样频率
        ra (float): 赤经
        dec (float): 赤纬
        psi (float): 极化角
        det (str): 探测器名称
        verbose (bool): 是否打印信息
    
    Returns:
        tuple: (new_ht, new_hp, new_hc) 探测器响应后的波形
    """
    # 时间向量
    tvec = np.arange(len(hp)) / float(fs)

    # 计算天线响应
    resp = AntennaResponse(det, ra, dec, psi, scalar=True, vector=True, times=0.0)
    Fp = resp.plus
    Fc = resp.cross
    ht = hp * Fp + hc * Fc

    # 计算地心时间延迟
    frDetector = lalsimulation.DetectorPrefixToLALDetector(det)
    tdelay = lal.TimeDelayFromEarthCenter(frDetector.location, ra, dec, 0.0)
    if verbose:
        print('{}: {} 地心时间延迟 = {}'.format(time.asctime(), det, tdelay))

    # 插值得到时间延迟后的信号
    ht_tck = interpolate.splrep(tvec, ht, s=0)
    hp_tck = interpolate.splrep(tvec, hp, s=0)
    hc_tck = interpolate.splrep(tvec, hc, s=0)
    tnew = tvec + tdelay
    new_ht = interpolate.splev(tnew, ht_tck, der=0, ext=1)
    new_hp = interpolate.splev(tnew, hp_tck, der=0, ext=1)
    new_hc = interpolate.splev(tnew, hc_tck, der=0, ext=1)

    return new_ht, new_hp, new_hc


def sim_data(fs, T_obs, snr=1.0, dets=['H1'], Nnoise=25, size=1000,
             mdist='astro', beta=[0.75, 0.95], verbose=True):
    """
    模拟完整的训练/测试数据集。
    
    生成包含噪声样本和信号样本的完整数据集。
    信号样本通过叠加BNS信号和不同噪声实现来增强数据。
    
    Args:
        fs (int): 采样频率(Hz)
        T_obs (float): 观测时间(秒)
        snr (float): 目标信噪比
        dets (list): 探测器列表
        Nnoise (int): 每个信号的噪声实现数
        size (int): 总样本数
        mdist (str): 质量分布类型
        beta (list): 峰值位置参数
        verbose (bool): 是否打印信息
    
    Returns:
        tuple: ([ts, yval], par)
            - ts (numpy.ndarray): 时序数据 (size, ndet, N)
            - yval (numpy.ndarray): 标签 (0=噪声, 1=信号)
            - par (list): 参数对象列表
    """
    yval = []
    ts = []
    par = []
    nclass = 2
    npclass = int(size / float(nclass))
    ndet = len(dets)
    psds = [gen_psd(fs, T_obs, op='AdvDesign', det=d) for d in dets]

    # 生成纯噪声样本（类别0）
    for x in xrange(npclass):
        if verbose:
            print('{}: 生成纯噪声样本'.format(time.asctime()))
        ts_new = np.array([gen_noise(fs, T_obs, psd.data.data) 
                          for psd in psds]).reshape(ndet, -1)
        ts.append(np.array([whiten_data(t, T_obs, fs, psd.data.data) 
                           for t, psd in zip(ts_new, psds)]).reshape(ndet, -1))
        par.append(None)
        yval.append(0)
        if verbose:
            print('{}: 完成 {}/{} 噪声样本'.format(time.asctime(), x + 1, npclass))

    # 生成信号样本（类别1）
    cnt = npclass
    while cnt < size:
        # 生成新的参数和波形
        par_new = gen_par(fs, T_obs, mdist=mdist, beta=beta, verbose=verbose)
        ts_new, _, _ = gen_bns(fs, T_obs, psds, snr=snr, dets=dets,
                               beta=beta, par=par_new, verbose=verbose)

        # 多个噪声实现
        for j in xrange(Nnoise):
            ts_noise = np.array([gen_noise(fs, T_obs, psd.data.data) 
                                for psd in psds]).reshape(ndet, -1)
            ts.append(np.array([whiten_data(t, T_obs, fs, psd.data.data) 
                               for t, psd in zip(ts_noise + ts_new, psds)]).reshape(ndet, -1))
            par.append(par_new)
            yval.append(1)
            cnt += 1
        if verbose:
            print('{}: 完成 {}/{} 信号样本'.format(
                time.asctime(), cnt - npclass, int(size / 2)))

    # 截断到目标长度
    ts = np.array(ts)[:size]
    yval = np.array(yval)[:size]
    par = par[:size]

    # 随机打乱数据
    idx = np.random.permutation(size)
    temp = [par[i] for i in idx]
    return [ts[idx], yval[idx]], temp


def main():
    """
    主函数：生成训练、验证和测试样本。
    
    命令行入口，解析参数并生成数据文件。
    """
    snr_mn = 0.0
    snr_cnt = 0

    # 解析命令行参数
    args = parser()
    if args.seed > 0:
        np.random.seed(args.seed)
    safeTobs = safe * args.Tobs

    # 分块生成数据
    nblock = int(np.ceil(float(args.Nsamp) / float(args.Nblock)))
    for i in xrange(nblock):
        print('{}: 开始生成BNS数据'.format(time.asctime()))
        ts, par = sim_data(args.fsample, safeTobs, args.snr, args.detectors,
                          args.Nnoise, size=args.Nblock, mdist=args.mdist,
                          beta=[0.75, 0.95])
        print('{}: 完成BNS数据生成 {}/{}'.format(time.asctime(), i + 1, nblock))

        # 保存时序数据
        f = open(args.basename + '_ts_' + str(i) + '.sav', 'wb')
        cPickle.dump(ts, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print('{}: 时序数据已保存'.format(time.asctime()))

        # 保存参数数据
        f = open(args.basename + '_params_' + str(i) + '.sav', 'wb')
        cPickle.dump(par, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print('{}: 参数数据已保存'.format(time.asctime()))

    print('{}: 成功完成'.format(time.asctime()))


if __name__ == "__main__":
    exit(main())
