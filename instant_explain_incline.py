# =====================================================================
# 1. 导入python模块
# =====================================================================
import math
import os
import csv
import pandas as pd
from iapws import IAPWS97
import matplotlib.pyplot as plt
# 图片字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 2. 文件输出位置设置
# =====================================================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_filename = os.path.join(base_dir, "result", "steam_injection_vertical_output.csv")
plot_filename = os.path.join(base_dir, "result","wellbore_trajectory.png")
f_out = open(output_filename, "w", encoding="utf-8")
print(f"初始化成功，结果将自动保存至:\n文本: {output_filename}\n图片: {plot_filename}")

# =====================================================================
# 3. 初始参数与已知条件设置
# =====================================================================
# --- 蒸汽与模拟参数 ---
T0_C = 320.0          # 蒸汽入口温度（℃）
m0 = 3.4              # 蒸汽质量流速（kg/s）
x0 = 0.9              # 蒸汽入口干度（0~1）
L = 2500            # 井筒测量总长度 MD (m)
dz = 10.0             # 计算步长（m）
g = 9.81      # 重力加速度 (m/s^2)  
max_sim_days = 200.0  # 总模拟时长 (天)
target_times = [0.05, 0.1, 1.0, 10.0, 50.0, 100.0, 150.0, 200.0] # 记录并在表格中展示的时间节点 (天)

# --- 海水段与地层环境参数 ---
L_sea = 100.0         # 海水段总垂深 TVD (m)
T_sea_surface = 20.0  # 海洋表面的海水温度 (℃)
T_sea_bed = 4.0       # 海床底部的海水温度 (℃)
Tt_Cm = 0.03          # 海床以下岩石地层的地温梯度 (℃/m)
R_sea_conv = 0.005    # 海水段极小的对流热阻

# --- 径向各组件尺寸参数 ---
r_ti, r_to = 0.038, 0.044          # 油管的内半径、外半径 (m)
r_ci, r_co = 0.076, 0.089          # 套管的内半径、外半径 (m)
r_w  = 0.108          # 水泥环外缘半径，即实际钻头打出的井眼半径 (m)
D = r_ti * 2          # 蒸汽流动的实际通道管径 (m)
A_tubing = math.pi * (r_ti**2) # 蒸汽流动的实际横截面积 (m^2)

# --- 径向各层等效热容与稳态热阻 ---
C_t, C_a, C_c, C_cem = 8000.0, 4000.0, 15000.0, 18000.0       # 油管、环空、套管、水泥环层热容 (J/(m·K))
R_st, R_ta, R_ac, R_cf = 0.01, 0.20, 0.05, 0.05          # 各层之间的稳态传热热阻 (m·K/W)，其中 R_ta=0.20 为隔热关键


alpha_e, lambda_e = 1.0e-6, 2.0        # 地层岩石的热扩散系数(m^2/s)与导热系数(W/(m·K))

# 井眼轨迹函数：计算倾角
def get_inclination_angle(MD):
    """
    0~600m: 垂直段 (90度)
    600~1500m: 造斜段 (角度从 90度 均匀降到 0度)
    """
    if MD <= 600.0:
        theta_deg = 90.0 # 0~600m 垂直 (90度)
    elif MD <= 1500.0:
        # 600~1500m 之间为造斜段，角度从 90度 线性均匀递减到 0度
        theta_deg = 90.0
    else:
        # 超过 1500m 进入纯水平段 (0度)
        theta_deg = 90.0
    # 返回角度 (用于表格展示) 和 弧度 (用于代码计算)
    return theta_deg, math.radians(theta_deg)

# =====================================================================
# 4. 理论模型函数
# =====================================================================
    """
    【Beggs-Brill 多相流管流模型】：负责计算任意倾角下的气液两相摩阻与持液率
    """
def calc_beggs_brill(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g, D, sigma, theta_rad):
    v_m = v_sl + v_sg # 气相与液相的表观速度之和，即混合物流速
    if v_m <= 1e-6: return 1.0, 0.02, rho_l # 容错：防止停井时速度为0导致的除零报错
    lambda_L = max(v_sl / v_m, 1e-5) # 无滑脱液相体积含量 (水占总流量的体积比)
    if lambda_L > 0.999: return 1.0, 0.02, rho_l  # 如果全是水，直接返回纯水参数
    # 计算无量纲准数
    N_Fr = v_m**2 / (g * D) # 弗劳德数，反映惯性力与重力的比值
    N_Lv = v_sl * ((rho_l / (g * sigma))**0.25) if sigma > 0 else 0 # 液相速度无量纲数
    
    # 确定 Beggs-Brill 流型分界线
    L1 = max(316 * (lambda_L**0.302), 1e-5)
    L2 = max(0.0009252 * (lambda_L**-2.4684), 1e-5)
    L3 = max(0.10 * (lambda_L**-1.4516), 1e-5)
    L4 = max(0.5 * (lambda_L**-6.738), 1e-5)

    # 判别当前处于哪种流型，并计算水平管的基础持液率 H_l0
    if (lambda_L < 0.01 and N_Fr < L1) or (lambda_L >= 0.01 and N_Fr < L2): 
        H_l0 = (0.98 * lambda_L**0.4846) / (N_Fr**0.0868)# 分离流
    elif (lambda_L >= 0.01 and L2 <= N_Fr <= L3):
        H_l0_s = (0.98 * lambda_L**0.4846) / (N_Fr**0.0868)
        H_l0_i = (0.845 * lambda_L**0.5351) / (N_Fr**0.0173)
        A = (L3 - N_Fr) / (L3 - L2)
        H_l0 = A * H_l0_s + (1 - A) * H_l0_i # 过渡流
    elif (0.01 <= lambda_L < 0.4 and L3 < N_Fr <= L1) or (lambda_L >= 0.4 and L3 < N_Fr <= L4):
        H_l0 = (0.845 * lambda_L**0.5351) / (N_Fr**0.0173)# 间歇流/段塞流
    else:
        H_l0 = (1.065 * lambda_L**0.5824) / (N_Fr**0.0609)# 分布流/泡状流

    # Beggs-Brill 独家的倾斜管角度修正模块
    e_val, f_val, g_val, h_val = 4.7, -0.3692, 0.1244, -0.5056 # 经验参数
    arg = e_val * (lambda_L**f_val) * (N_Lv**g_val) * (N_Fr**h_val)
    C_corr = (1 - lambda_L) * math.log(arg) if arg > 0 else 0
    C_corr = max(C_corr, 0)
    
    # 结合当前深度的井斜角(theta_rad)，计算角度修正乘子 psi
    # 注：Beggs-Brill定义向下流为负角度，因此传入 -theta_rad
    sin_val = math.sin(1.8 * (-theta_rad)) # 倾角修正
    psi = 1 + C_corr * (sin_val - 0.333 * (sin_val**3))

    # 得到真实的截面持液率 H_l (截面里真正有多少比例是水)
    H_l = max(0.0, min(1.0, H_l0 * psi)) 

    # 计算混合物的真实密度 (算重力压降用)    
    rho_m = rho_l * H_l + rho_g * (1 - H_l)

    # 计算无滑脱状态下的密度和黏度 (算摩阻压降用)
    rho_ns = rho_l * lambda_L + rho_g * (1 - lambda_L)
    mu_ns = mu_l * lambda_L + mu_g * (1 - lambda_L)
    Re_ns = rho_ns * v_m * D / mu_ns # 无滑脱雷诺数

    # 计算基础平滑管的摩擦系数 f_ns    
    f_ns = 0.3164 / (Re_ns**0.25) if Re_ns > 2300 else 64.0 / max(Re_ns, 1e-5)
    
    # 计算气液两相相互作用对摩擦系数的放大效应指数 S
    y = lambda_L / max(H_l**2, 1e-5)
    if 1.0 < y < 1.2: S = math.log(2.2 * y - 1.2)
    else:
        ln_y = math.log(y) if y > 0 else -10
        denom = -0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4
        if abs(denom) < 1e-5: denom = 1e-5
        S = ln_y / denom
    return H_l, f_ns * math.exp(S), rho_m # 返回: 真实持液率, 最终摩阻系数, 真实混合密度

def calc_hasan_kabir_Re(t_sec, r_co, alpha_e, lambda_e):
    """
    【瞬态地层热阻模型 (Hasan-Kabir)】：模拟周围岩石被逐渐“捂热”的过程。
    时间越长，热阻越大，热量流失越慢。
    """
    if t_sec <= 0: return 0.0
    t_D = (alpha_e * t_sec) / (r_co**2) # 无量纲时间
    if t_D <= 1.5: return (1.1281 * math.sqrt(t_D)) / (2 * math.pi * lambda_e)
    elif t_D <= 10.0: return ((0.4063 + 0.5 * math.log(t_D)) * (1 + 0.6 / t_D)) / (2 * math.pi * lambda_e)
    else: return (0.5 * math.log(t_D) + 0.403) / (2 * math.pi * lambda_e)

def thomas_algorithm(a, b, c, d):
    """
    【追赶法 TDMA 求解器】：用于极速求解 4 层井筒径向传热形成的三对角矩阵。
    """    
    n = len(d)
    c_prime, d_prime, x = [0.0]*n, [0.0]*n, [0.0]*n
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / m if i < n - 1 else 0.0
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1): x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return x 

# =====================================================================
# 5. 初始化网格与环境温度
# =====================================================================
# 根据给定的入口温度 320℃ 和干度 0.9，查表获取初始压力和焓值
T0_K = T0_C + 273.15
inlet_state = IAPWS97(T=T0_K, x=x0)
P0_Pa = inlet_state.P * 1e6
h0_Jkg = inlet_state.h * 1000

N_z = int(L / dz) # 计算一共要切分成多少个空间网格段
Tt_prev, Ta_prev, Tc_prev, Tcem_prev = [], [], [], [] # 建立上一时刻4层温度记忆库

TVD_init = 0.0
for step in range(N_z):
    z_MD = step * dz
    # 根据轨迹获取当前微元段中点的倾斜角
    _, theta_rad = get_inclination_angle(z_MD + dz / 2.0)
    # 利用三角函数计算中点的真实垂深 (TVD)
    TVD_mid = TVD_init + (dz / 2.0) * math.sin(theta_rad)
    

    if TVD_mid <= L_sea: 
        # 处于 0-100m 海水段：环境温度受海水深度影响递减
        T_f_C = T_sea_surface - (T_sea_surface - T_sea_bed) * (TVD_mid / L_sea)
    else: 
        # 处于 >100m 岩石段：环境温度随地温梯度上升
        T_f_C = T_sea_bed + Tt_Cm * (TVD_mid - L_sea)
        
    Tf_K = T_f_C + 273.15 # 将计算出的环境温度作为 t=0 时刻各井筒组件的初始温度
    Tt_prev.append(Tf_K); Ta_prev.append(Tf_K); Tc_prev.append(Tf_K); Tcem_prev.append(Tf_K)
    TVD_init += dz * math.sin(theta_rad) # 为计算下一个网格的垂深做准备

# =====================================================================
# 6. 核心时空双重推演
# =====================================================================
all_results = []
current_time_days = 0.0
print(f"启动径向多节点(含海水段与定向井轨迹)推演...", file=f_out)

while current_time_days < max_sim_days - 1e-6:
    # 注汽极早期采用极小的时间步捕捉瞬间温度骤降
    if current_time_days < 1.0: dt_days = 0.05
    elif current_time_days < 10.0: dt_days = 0.5
    elif current_time_days < 50.0: dt_days = 2.0
    elif current_time_days < 100.0: dt_days = 5.0
    else: dt_days = 10.0

    # 确保时间在我们想要抓取输出的时间节点上    
    next_targets = [t for t in target_times if t > current_time_days + 1e-6]
    next_target = next_targets[0] if next_targets else max_sim_days
    if current_time_days + dt_days > next_target: dt_days = next_target - current_time_days
    if dt_days < 1e-6: dt_days = 1e-6
        
    current_time_days += dt_days
    current_t_sec = current_time_days * 24 * 3600

    # 计算当前时间点，地层岩石提供的瞬态热阻
    R_e_t_formation = calc_hasan_kabir_Re(current_t_sec, r_w, alpha_e, lambda_e)
    
    # 为一整条井筒的空间推演做变量准备
    P_current, h_current = P0_Pa, h0_Jkg
    dP_guess, dh_guess = 80000.0, -10000.0 # 每空间步长的初始压降和焓降猜测值
    
    # 初始化用于存放该时刻整井剖面数据的列表
    z_list_t, TVD_list_t = [0.0], [0.0]
    theta_deg_list_t = [90.0]  # 井口绝对是90度垂直
    
    P_list_t, h_list_t, x_list_t, Ts_list_t = [P0_Pa/1e6], [h0_Jkg/1000], [x0], [T0_C]
    Tt_list_t, Ta_list_t, Tc_list_t, Tcem_list_t = [T0_C], [T0_C], [T0_C], [T0_C]
    Tf_list_t = [T_sea_surface] 
    
    Tt_curr, Ta_curr, Tc_curr, Tcem_curr = [0.0]*N_z, [0.0]*N_z, [0.0]*N_z, [0.0]*N_z
    TVD_current = 0.0 
    
    # ---> 开始沿着井深 (Z轴) 往下逐段推演
    for step in range(N_z):
        z_current = step * dz
        
        # 获取当前深度的倾角，累加TVD
        theta_deg, theta_rad = get_inclination_angle(z_current + dz / 2.0)
        TVD_mid = TVD_current + (dz / 2.0) * math.sin(theta_rad)
        
        # 判断海水段还是地层段
        if TVD_mid <= L_sea:
            T_f_C = T_sea_surface - (T_sea_surface - T_sea_bed) * (TVD_mid / L_sea)
            # 海水段的外部阻力极小，且无水泥环缓存热量
            R_cf_curr, R_f_curr, C_cem_curr = R_sea_conv, R_sea_conv, C_c         
        else:
            T_f_C = T_sea_bed + Tt_Cm * (TVD_mid - L_sea)
            # 岩石段具备水泥环稳态热阻和岩石瞬态热阻
            R_cf_curr, R_f_curr, C_cem_curr = R_cf, R_cf + R_e_t_formation, C_cem       
            
        T_f_K = T_f_C + 273.15

        # ---> 内部迭代：联合求解当前段的真实压力与焓值
        for iteration in range(100):
            # 取微元段中点的平均压力和平均比焓
            P_avg, h_avg = P_current + dP_guess / 2.0, h_current + dh_guess / 2.0
            try:
                # 查表获取水和蒸汽的饱和参数界限
                sat_l, sat_v = IAPWS97(P=P_avg/1e6, x=0), IAPWS97(P=P_avg/1e6, x=1)
            except BaseException: break
                
            hl_Jkg, hv_Jkg = sat_l.h * 1000, sat_v.h * 1000
            T_avg_K = sat_l.T  # 蒸汽处于饱和态，温度完全由压力决定

            # 利用能量守恒算出真实的蒸汽干度 x
            x_avg = max(0.0, min(1.0, (h_avg - hl_Jkg) / (hv_Jkg - hl_Jkg)))

            # 根据干度和密度算出气、液两相流速
            v_sg, v_sl = (m0 * x_avg) / (sat_v.rho * A_tubing), (m0 * (1.0 - x_avg)) / (sat_l.rho * A_tubing)
            
            # 引入 Beggs-Brill 算得真实倾角下的摩阻
            H_l, f_m, rho_m = calc_beggs_brill(v_sl, v_sg, sat_l.rho, sat_v.rho, sat_l.mu, sat_v.mu, D, sat_l.sigma, theta_rad)
            
            # 计算重力(倾斜角)及摩擦带来的压力降低
            dP_calc = (rho_m * g * math.sin(theta_rad) - f_m * rho_m * (v_sl + v_sg)**2 / (2 * D)) * dz
            
            # 准备组装 4 层径向传热矩阵
            dt_step = dt_days * 24 * 3600
            B_t, B_a, B_c, B_cem = C_t/dt_step, C_a/dt_step, C_c/dt_step, C_cem_curr/dt_step
            
            # 油管层系数 (主受内侧高温蒸汽和自己热惯性的影响)
            b0 = (1.0/R_st) + B_t + (1.0/R_ta)
            c0 = - (1.0/R_ta)
            d0 = B_t * Tt_prev[step] + (1.0/R_st) * T_avg_K
            
            # 环空层系数
            a1, b1, c1 = - (1.0/R_ta), (1.0/R_ta) + B_a + (1.0/R_ac), - (1.0/R_ac)
            d1 = B_a * Ta_prev[step]
            
            # 套管层系数
            a2, b2, c2 = - (1.0/R_ac), (1.0/R_ac) + B_c + (1.0/R_cf_curr), - (1.0/R_cf_curr)
            d2 = B_c * Tc_prev[step]
            
            # 水泥/隔水管外壁层系数 (受外部冷环境影响极大)
            a3, b3 = - (1.0/R_cf_curr), (1.0/R_cf_curr) + B_cem + (1.0/R_f_curr)
            d3 = B_cem * Tcem_prev[step] + (1.0/R_f_curr) * T_f_K
            
            # 利用追赶法瞬间解出 4 层温度
            T_res = thomas_algorithm([0.0, a1, a2, a3], [b0, b1, b2, b3], [c0, c1, c2, 0.0], [d0, d1, d2, d3])
            Tt_val, Ta_val, Tc_val, Tcem_val = T_res[0], T_res[1], T_res[2], T_res[3]
            
            # 根据最新算出的油管内壁温度 (Tt_val)，计算真实向外散失的热量
            q_loss_per_m = (T_avg_K - Tt_val) / R_st

            # 计算散热及动能转化带来的比焓下降
            dh_calc = (g * math.sin(theta_rad) - q_loss_per_m / m0) * dz
            
            # 检查收敛误差
            err_P = abs((dP_calc - dP_guess) / dP_guess) if dP_guess != 0 else abs(dP_calc)
            err_h = abs((dh_calc - dh_guess) / dh_guess) if dh_guess != 0 else abs(dh_calc)
            
            if err_P < 1e-4 and err_h < 1e-4: break
            else: dP_guess, dh_guess = 0.5 * dP_guess + 0.5 * dP_calc, 0.5 * dh_guess + 0.5 * dh_calc

        # 将算好的数据存入本时刻的网格记录表        
        Tt_curr[step], Ta_curr[step], Tc_curr[step], Tcem_curr[step] = Tt_val, Ta_val, Tc_val, Tcem_val
        P_current, h_current = P_current + dP_calc, h_current + dh_calc
        dP_guess, dh_guess = dP_calc, dh_calc 
        
        TVD_current += dz * math.sin(theta_rad) # 更新垂深
        
        # 将各类物理参数追加到输出队列
        z_list_t.append(z_current + dz)
        TVD_list_t.append(TVD_current)
        theta_deg_list_t.append(theta_deg) # 记录角度
        P_list_t.append(P_current / 1e6)
        h_list_t.append(h_current / 1000)
        x_list_t.append(x_avg)
        Ts_list_t.append(T_avg_K - 273.15)
        Tt_list_t.append(Tt_val - 273.15)
        Ta_list_t.append(Ta_val - 273.15)
        Tc_list_t.append(Tc_val - 273.15)
        Tcem_list_t.append(Tcem_val - 273.15)
        Tf_list_t.append(T_f_C) 

    # 完成一整条井筒的推演，把当下的温度记忆为“上一时刻”，供下一秒计算
    Tt_prev, Ta_prev = Tt_curr, Ta_curr
    Tc_prev, Tcem_prev = Tc_curr, Tcem_curr
    
    # 若抵达目标抓取时间，则生成 DataFrame 表格存入集合
    is_target = any(abs(current_time_days - t_target) < 1e-5 for t_target in target_times)
    if is_target:
        print(f" -> 已记录: {current_time_days:.2f} 天")
        df_t = pd.DataFrame({
            "时间(天)": [round(current_time_days, 2)] * len(z_list_t),
            "测深MD(m)": z_list_t,
            "垂深TVD(m)": [round(v, 2) for v in TVD_list_t],
            "倾角(度)": [round(v, 2) for v in theta_deg_list_t], # [★★★ 新增输出列 ★★★]
            "环境温(℃)": Tf_list_t,
            "压力(MPa)": P_list_t,
            "比焓(kJ/kg)": h_list_t,
            "干度": x_list_t,
            "蒸汽温(℃)": Ts_list_t,
            "油管温(℃)": Tt_list_t,
            "环空温(℃)": Ta_list_t,
            "套管温(℃)": Tc_list_t,
            "第四层温(℃)": Tcem_list_t,
        })
        all_results.append(df_t)

# =====================================================================
# 7. 导出数据与生成井筒示意图
# =====================================================================
df_results = pd.concat(all_results, ignore_index=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n=== 海上热采井：含海水段+定向轨迹 全剖面精细化推演 ===", file=f_out)
print(df_results, file=f_out)
f_out.close()

df_results.to_csv(output_filename, index=False, encoding='utf-8-sig', 
                  float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

# 利用 matplotlib 绘制井筒轨迹示意图
print(f"正在绘制井筒轨迹示意图...")

# 1. 计算水平位移 (X) 以供画图
MD_plot, TVD_plot, X_plot = [0.0], [0.0], [0.0]
current_TVD, current_X = 0.0, 0.0

for step in range(N_z):
    z_MD = step * dz
    _, theta_rad = get_inclination_angle(z_MD + dz / 2.0)
    current_TVD += dz * math.sin(theta_rad)
    current_X += dz * math.cos(theta_rad) # 水平位移 = 步长 * cos(倾角)
    MD_plot.append(z_MD + dz)
    TVD_plot.append(current_TVD)
    X_plot.append(current_X)

# 2. 建立画布
plt.figure(figsize=(10, 8))

# 3. 绘制背景分层色块 (海水 vs 地层)
max_tvd = max(TVD_plot) + 50
plt.axhspan(0, L_sea, color='#e0f7fa', alpha=0.8, label=f'海水段 (0-{L_sea}m)')
plt.axhspan(L_sea, max_tvd, color='#fbe9e7', alpha=0.6, label='岩石地层段')
plt.axhline(y=L_sea, color='blue', linestyle='--', linewidth=1, alpha=0.5)

# 4. 绘制井筒真实轨迹
plt.plot(X_plot, TVD_plot, color='black', linewidth=4, label='井筒轨迹 (Steam Injection Well)')

# 5. 标注关键转折点
plt.scatter([X_plot[0]], [TVD_plot[0]], color='red', s=100, zorder=5, label='海上平台 (井口)')
# 找到造斜点 (600m MD 处)
kop_idx = int(600 / dz)
if kop_idx < len(X_plot):
    plt.scatter([X_plot[kop_idx]], [TVD_plot[kop_idx]], color='orange', s=80, zorder=5)
    plt.text(X_plot[kop_idx]+20, TVD_plot[kop_idx], '造斜点 (MD:600m)', verticalalignment='center')

# 6. 图表样式与输出
plt.gca().invert_yaxis() # 重点：深度坐标轴必须反转，越往下数值越大
plt.xlabel('水平位移 / Horizontal Displacement (m)', fontsize=12)
plt.ylabel('真实垂深 / True Vertical Depth TVD (m)', fontsize=12)
plt.title('海上稠油热采：大位移井筒轨迹与环境分层示意图', fontsize=16, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower left', fontsize=11)

plt.tight_layout()
plt.savefig(plot_filename, dpi=300) # 输出井轨迹图片
plt.close()

print(f"运算完成！")