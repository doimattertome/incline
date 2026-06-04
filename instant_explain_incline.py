# =====================================================================
# 1. 导入python模块
# =====================================================================
import math
import os
import csv
import pandas as pd
from iapws import IAPWS97

# =====================================================================
# 2. 文件输出位置设置
# =====================================================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_filename = os.path.join(base_dir, "result", "steam_injection_vertical_output.csv")
# 若目录不存在则创建
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
f_out = open(output_filename, "w", encoding="utf-8")
print(f"初始化成功，结果将自动保存至:\n文本: {output_filename}")

# =====================================================================
# 3. 初始参数与已知条件设置
# =====================================================================
# --- 蒸汽与模拟参数 ---
T0_C = 320.0          # 蒸汽入口温度（℃）
m0 = 3.4              # 蒸汽质量流速（kg/s）
x0 = 0.9              # 蒸汽入口干度（0~1）
L = 3000              # 井筒测量总长度 MD (m)
dz = 10               # 计算步长（m）
g = 9.81              # 重力加速度 (m/s^2)  
max_sim_days = 15.0   # 总模拟时长 (天) 【修改：15天常规热采注汽】
target_times = [0.1, 1.0, 5.0, 15.0] # 记录并在表格中展示的时间节点 (天) 【修改：增加演化观察节点】

# 稳态/瞬态切换开关
is_steady_state = False # True 为计算稳态(热容失效)，False 为计算瞬态(有热惯性) 【修改：开启瞬态推演】

# --- 海水段与地层环境参数 ---
L_sea = 500.0         # 海水段总垂深 TVD (m) 【修改：设置500m海水段】
T_sea_surface = 20.0  # 海洋表面的海水温度 (℃)
T_sea_bed = 4.0       # 海床底部的海水温度 (℃)
Tt_Cm = 0.03          # 海床以下岩石地层的地温梯度 (℃/m)                                                                                                    
R_sea_conv = 0.005    # 海水段极小的对流热阻

# --- 径向各组件尺寸参数与粗糙度【修改点 1】 ---
r_ti, r_to = 0.038, 0.057          # 油管的内半径、外半径 (m)
r_ci, r_co = 0.11, 0.12            # 套管的内半径、外半径 (m)
r_w  = 0.13                        # 水泥环外缘半径，即实际钻头打出的井眼半径 (m)
D = r_ti * 2                       # 蒸汽流动的实际通道管径 (m)
A_tubing = math.pi * (r_ti**2)     # 蒸汽流动的实际横截面积 (m^2)
epsilon = 4.5e-5                   # 油管内壁绝对粗糙度 (m) (商业钢管通常取0.045mm)

# --- 径向各层等效热容与热阻【修改点 2：接箍热桥修正】 ---                
C_t, C_a, C_c, C_cem = 8000.0, 4000.0, 15000.0, 18000.0  # 油管、环空、套管、水泥环层热容 (J/(m·K))

L_t = 9.6           # 单根隔热油管组合总长 (m)
L_c = 0.18          # 接箍有效传热长度 (m)
R_st_body = 0.01    # 纯管身真空绝热稳态热阻 (m·K/W)
R_st_cpl = 0.002    # 接箍处裸露金属局部热阻 (m·K/W) (极小值，表征热桥)

# 计算轴向长度加权综合等效热阻 R_st_eq
inv_R_st_eq = (1.0 - L_c / L_t) * (1.0 / R_st_body) + (L_c / L_t) * (1.0 / R_st_cpl)
R_st_eq = 1.0 / inv_R_st_eq
print(f"-> 考虑接箍热桥效应后，油管综合等效热阻修正为: {R_st_eq:.5f} m·K/W (管身纯热阻为 {R_st_body})")

R_ac, R_cf = 0.05, 0.05  # 各层之间的稳态传热热阻 (m·K/W) (环空热阻现改为动态计算)

alpha_e, lambda_e = 1.0e-6, 2.0        # 地层岩石的热扩散系数(m^2/s)与导热系数(W/(m·K))

# 井眼轨迹函数：计算倾角
def get_inclination_angle(MD):
    """
    0~600m: 垂直段 (90度)
    600~1500m: 造斜段 (角度从 90度 均匀降到 0度)
    """
    if MD <= 600.0:
        theta_deg = 90.0
    elif MD <= 1500.0:
        theta_deg = 90.0
    else:
        theta_deg = 90.0
    return theta_deg, math.radians(theta_deg)

# =====================================================================
# 4. 理论模型函数
# =====================================================================
def calc_beggs_brill(v_sl, v_sg, rho_l, rho_g, mu_l, mu_g, D, sigma, theta_rad, epsilon):
    """
    【Beggs-Brill 多相流管流模型】：负责计算任意倾角下的气液两相摩阻与持液率
    """
    v_m = v_sl + v_sg # 气相与液相的表观速度之和，即混合物流速
    if v_m <= 1e-6: return 1.0, 0.02, rho_l # 容错
    lambda_L = max(v_sl / v_m, 1e-5) # 无滑脱液相体积含量
    if lambda_L > 0.999: return 1.0, 0.02, rho_l  
    
    # 计算无量纲准数
    N_Fr = v_m**2 / (g * D) 
    N_Lv = v_sl * ((rho_l / (g * sigma))**0.25) if sigma > 0 else 0 
    
    # 确定 Beggs-Brill 流型分界线
    L1 = max(316 * (lambda_L**0.302), 1e-5)
    L2 = max(0.0009252 * (lambda_L**-2.4684), 1e-5)
    L3 = max(0.10 * (lambda_L**-1.4516), 1e-5)
    L4 = max(0.5 * (lambda_L**-6.738), 1e-5)

    # 判别流型
    if (lambda_L < 0.01 and N_Fr < L1) or (lambda_L >= 0.01 and N_Fr < L2): 
        H_l0 = (0.98 * lambda_L**0.4846) / (N_Fr**0.0868)
    elif (lambda_L >= 0.01 and L2 <= N_Fr <= L3):
        H_l0_s = (0.98 * lambda_L**0.4846) / (N_Fr**0.0868)
        H_l0_i = (0.845 * lambda_L**0.5351) / (N_Fr**0.0173)
        A = (L3 - N_Fr) / (L3 - L2)
        H_l0 = A * H_l0_s + (1 - A) * H_l0_i
    elif (0.01 <= lambda_L < 0.4 and L3 < N_Fr <= L1) or (lambda_L >= 0.4 and L3 < N_Fr <= L4):
        H_l0 = (0.845 * lambda_L**0.5351) / (N_Fr**0.0173)
    else:
        H_l0 = (1.065 * lambda_L**0.5824) / (N_Fr**0.0609)

    # 倾斜管角度修正模块
    e_val, f_val, g_val, h_val = 4.7, -0.3692, 0.1244, -0.5056
    arg = e_val * (lambda_L**f_val) * (N_Lv**g_val) * (N_Fr**h_val)
    C_corr = (1 - lambda_L) * math.log(arg) if arg > 0 else 0
    C_corr = max(C_corr, 0)
    
    sin_val = math.sin(1.8 * (-theta_rad))
    psi = 1 + C_corr * (sin_val - 0.333 * (sin_val**3))

    H_l = max(0.0, min(1.0, H_l0 * psi)) 
    rho_m = rho_l * H_l + rho_g * (1 - H_l)

    rho_ns = rho_l * lambda_L + rho_g * (1 - lambda_L)
    mu_ns = mu_l * lambda_L + mu_g * (1 - lambda_L)
    Re_ns = rho_ns * v_m * D / mu_ns 

    # 【修改点 3：对齐理论，引入粗糙度与 Swamee-Jain 方程】
    if Re_ns > 2300:
        # 湍流状态：使用考虑绝对粗糙度的 Swamee-Jain 方程
        denom = math.log10(epsilon / (3.7 * D) + 5.74 / (Re_ns**0.9))
        f_ns = 0.25 / (denom**2)
    else:
        # 层流状态：使用经典 Poiseuille 方程
        f_ns = 64.0 / max(Re_ns, 1e-5)
    
    y = lambda_L / max(H_l**2, 1e-5)
    if 1.0 < y < 1.2: S = math.log(2.2 * y - 1.2)
    else:
        ln_y = math.log(y) if y > 0 else -10
        denom = -0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4
        if abs(denom) < 1e-5: denom = 1e-5
        S = ln_y / denom
    return H_l, f_ns * math.exp(S), rho_m 

def calc_hasan_kabir_Re(t_sec, r_co, alpha_e, lambda_e):
    """
    【瞬态地层热阻模型 (Hasan-Kabir)】
    """
    if t_sec <= 0: return 0.0
    t_D = (alpha_e * t_sec) / (r_co**2) 
    if t_D <= 1.5: return (1.1281 * math.sqrt(t_D)) / (2 * math.pi * lambda_e)
    elif t_D <= 10.0: return ((0.4063 + 0.5 * math.log(t_D)) * (1 + 0.6 / t_D)) / (2 * math.pi * lambda_e)
    else: return (0.5 * math.log(t_D) + 0.403) / (2 * math.pi * lambda_e)

def calc_annulus_R_ta(T_t_K, T_c_K, r_to, r_ci, P_annulus=101325.0):
    """
    【动态环空热阻】：计算环空自然对流与辐射的综合当量热阻 (m·K/W)
    """
    if abs(T_t_K - T_c_K) < 0.1:
        T_t_K += 0.1 
        
    T_avg = (T_t_K + T_c_K) / 2.0
    delta_T = abs(T_t_K - T_c_K)
    
    # 1. 氮气物性估算
    rho = P_annulus / (296.8 * T_avg)                     
    mu = 1.78e-5 * ((T_avg / 288.15)**0.75)               
    lambda_g = 0.025 * ((T_avg / 288.15)**0.8)            
    cp = 1040.0                                           
    
    nu = mu / rho                                         
    alpha = lambda_g / (rho * cp)                         
    beta = 1.0 / T_avg                                    
    
    # 2. 计算无量纲准数
    L_c = r_ci - r_to                                     
    Gr = (9.81 * beta * delta_T * (L_c**3)) / (nu**2)
    Pr = nu / alpha
    Ra = Gr * Pr
    
    # 3. 对流 Nusselt 数
    if Ra < 2000: Nu = 1.0 
    elif Ra < 6000: Nu = 0.119 * (Ra**0.227)
    elif Ra < 2e5: Nu = 0.153 * (Ra**0.25)
    else: Nu = 0.092 * (Ra**0.333)
        
    lambda_eq_conv = lambda_g * Nu 
    
    # 4. 辐射换热系数
    eps_t, eps_c = 0.8, 0.8 
    F_tc = 1.0 / (1.0/eps_t + (r_to/r_ci)*(1.0/eps_c - 1.0))
    hr = 5.67e-8 * F_tc * (T_t_K**2 + T_c_K**2) * (T_t_K + T_c_K)
    lambda_eq_rad = hr * r_to * math.log(r_ci / r_to)
    
    # 5. 综合热阻
    lambda_eq_total = lambda_eq_conv + lambda_eq_rad
    R_ta = math.log(r_ci / r_to) / (2 * math.pi * lambda_eq_total)
    
    return R_ta

def thomas_algorithm(a, b, c, d):
    """
    【追赶法 TDMA 求解器】
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
T0_K = T0_C + 273.15
inlet_state = IAPWS97(T=T0_K, x=x0)
P0_Pa = inlet_state.P * 1e6
h0_Jkg = inlet_state.h * 1000

N_z = int(L / dz) 
Tt_prev, Ta_prev, Tc_prev, Tcem_prev = [], [], [], [] 

TVD_init = 0.0
for step in range(N_z):
    z_MD = step * dz
    _, theta_rad = get_inclination_angle(z_MD + dz / 2.0)
    TVD_mid = TVD_init + (dz / 2.0) * math.sin(theta_rad)
    
    if TVD_mid <= L_sea: 
        T_f_C = T_sea_surface - (T_sea_surface - T_sea_bed) * (TVD_mid / L_sea)
    else: 
        T_f_C = T_sea_bed + Tt_Cm * (TVD_mid - L_sea)
        
    Tf_K = T_f_C + 273.15 
    Tt_prev.append(Tf_K); Ta_prev.append(Tf_K); Tc_prev.append(Tf_K); Tcem_prev.append(Tf_K)
    TVD_init += dz * math.sin(theta_rad) 

# =====================================================================
# 6. 核心时空双重推演
# =====================================================================
all_results = []
current_time_days = 0.0
print(f"启动径向多节点(含海水段与定向井轨迹)推演...", file=f_out)

while current_time_days < max_sim_days - 1e-6:
    # 【修改：时间步长动态控制，前期极短，后期变长以加速】
    if current_time_days < 0.1: dt_days = 0.01      # 前0.1天：径向吸热最剧烈，采用极小步长
    elif current_time_days < 1.0: dt_days = 0.05    # 0.1 ~ 1天：系统逐渐升温，较小步长
    elif current_time_days < 5.0: dt_days = 0.2     # 1 ~ 5天：温升变缓，中等步长
    else: dt_days = 0.5                             # 5天以后：逐渐逼近稳态，采用较大步长

    next_targets = [t for t in target_times if t > current_time_days + 1e-6]
    next_target = next_targets[0] if next_targets else max_sim_days
    if current_time_days + dt_days > next_target: dt_days = next_target - current_time_days
    if dt_days < 1e-6: dt_days = 1e-6
        
    current_time_days += dt_days
    current_t_sec = current_time_days * 24 * 3600
 
    if is_steady_state:
        # 稳态时，假设地层已经注汽极长时间（例如10年），地层热阻达到长期稳定常数
        steady_time_sec = 10.0 * 365 * 24 * 3600
        R_e_t_formation = calc_hasan_kabir_Re(steady_time_sec, r_w, alpha_e, lambda_e)
    else:
        R_e_t_formation = calc_hasan_kabir_Re(current_t_sec, r_w, alpha_e, lambda_e)
    
    P_current, h_current = P0_Pa, h0_Jkg
    dP_guess, dh_guess = 80000.0, -10000.0 
    
    z_list_t, TVD_list_t = [0.0], [0.0]
    theta_deg_list_t = [90.0]  
    
    P_list_t, h_list_t, x_list_t, Ts_list_t = [P0_Pa/1e6], [h0_Jkg/1000], [x0], [T0_C]
    Tt_list_t, Ta_list_t, Tc_list_t, Tcem_list_t = [T0_C], [T0_C], [T0_C], [T0_C]
    Tf_list_t = [T_sea_surface] 
    
    Tt_curr, Ta_curr, Tc_curr, Tcem_curr = [0.0]*N_z, [0.0]*N_z, [0.0]*N_z, [0.0]*N_z
    TVD_current = 0.0 
    
    for step in range(N_z):
        z_current = step * dz
        theta_deg, theta_rad = get_inclination_angle(z_current + dz / 2.0)
        TVD_mid = TVD_current + (dz / 2.0) * math.sin(theta_rad)
        
        if TVD_mid <= L_sea:
            T_f_C = T_sea_surface - (T_sea_surface - T_sea_bed) * (TVD_mid / L_sea)
            R_cf_curr, R_f_curr, C_cem_curr = R_sea_conv, R_sea_conv, C_c         
        else:
            T_f_C = T_sea_bed + Tt_Cm * (TVD_mid - L_sea)
            R_cf_curr, R_f_curr, C_cem_curr = R_cf, R_cf + R_e_t_formation, C_cem       
            
        T_f_K = T_f_C + 273.15
        
        # 内部迭代前，给出环空热阻的一个初始猜测值
        R_ta_curr = calc_annulus_R_ta(Tt_prev[step], Tc_prev[step], r_to, r_ci)

        for iteration in range(100):
            P_avg, h_avg = P_current + dP_guess / 2.0, h_current + dh_guess / 2.0
            try:
                sat_l = IAPWS97(P=P_avg/1e6, x=0)
                sat_v = IAPWS97(P=P_avg/1e6, x=1)
            except BaseException: break
                
            hl_Jkg, hv_Jkg = sat_l.h * 1000, sat_v.h * 1000
            
            if h_avg >= hl_Jkg:
                # 两相区 (含湿蒸汽)
                x_avg = min(1.0, (h_avg - hl_Jkg) / (hv_Jkg - hl_Jkg))
                T_avg_K = sat_l.T
                rho_l_val = sat_l.rho
            else:
                # 过冷水区 (全部冷凝为单相水)
                x_avg = 0.0
                try:
                    subcooled_water = IAPWS97(P=P_avg/1e6, h=h_avg/1000)
                    T_avg_K = subcooled_water.T
                    rho_l_val = subcooled_water.rho
                except BaseException:
                    T_avg_K = sat_l.T  
                    rho_l_val = sat_l.rho

            v_sg, v_sl = (m0 * x_avg) / (sat_v.rho * A_tubing), (m0 * (1.0 - x_avg)) / (rho_l_val * A_tubing)
            
            # 传入修正后的粗糙度参数 epsilon
            H_l, f_m, rho_m = calc_beggs_brill(v_sl, v_sg, rho_l_val, sat_v.rho, sat_l.mu, sat_v.mu, D, sat_l.sigma, theta_rad, epsilon)
            dP_calc = (rho_m * g * math.sin(theta_rad) - f_m * rho_m * (v_sl + v_sg)**2 / (2 * D)) * dz
            
            if is_steady_state:
                dt_step = 1e15 # 稳态时时间步长设为无穷大，消除管柱吸热
            else:
                dt_step = dt_days * 24 * 3600

            B_t, B_a, B_c, B_cem = C_t/dt_step, C_a/dt_step, C_c/dt_step, C_cem_curr/dt_step
            
            # 使用修正后的等效热阻 R_st_eq 替换原有的 R_st
            b0 = (1.0/R_st_eq) + B_t + (1.0/R_ta_curr)
            c0 = - (1.0/R_ta_curr)
            d0 = B_t * Tt_prev[step] + (1.0/R_st_eq) * T_avg_K
            
            a1, b1, c1 = - (1.0/R_ta_curr), (1.0/R_ta_curr) + B_a + (1.0/R_ac), - (1.0/R_ac)
            d1 = B_a * Ta_prev[step]
            
            a2, b2, c2 = - (1.0/R_ac), (1.0/R_ac) + B_c + (1.0/R_cf_curr), - (1.0/R_cf_curr)
            d2 = B_c * Tc_prev[step]
            
            a3, b3 = - (1.0/R_cf_curr), (1.0/R_cf_curr) + B_cem + (1.0/R_f_curr)
            d3 = B_cem * Tcem_prev[step] + (1.0/R_f_curr) * T_f_K
            
            T_res = thomas_algorithm([0.0, a1, a2, a3], [b0, b1, b2, b3], [c0, c1, c2, 0.0], [d0, d1, d2, d3])
            Tt_val, Ta_val, Tc_val, Tcem_val = T_res[0], T_res[1], T_res[2], T_res[3]
            
            # 动态更新环空热阻
            R_ta_curr = calc_annulus_R_ta(Tt_val, Tc_val, r_to, r_ci)

            # 同样使用修正后的等效热阻 R_st_eq 计算热损
            q_loss_per_m = (T_avg_K - Tt_val) / R_st_eq
            dh_calc = (g * math.sin(theta_rad) - q_loss_per_m / m0) * dz
            
            err_P = abs((dP_calc - dP_guess) / dP_guess) if dP_guess != 0 else abs(dP_calc)
            err_h = abs((dh_calc - dh_guess) / dh_guess) if dh_guess != 0 else abs(dh_calc)
            
            if err_P < 1e-4 and err_h < 1e-4: break
            else: dP_guess, dh_guess = 0.5 * dP_guess + 0.5 * dP_calc, 0.5 * dh_guess + 0.5 * dh_calc

        Tt_curr[step], Ta_curr[step], Tc_curr[step], Tcem_curr[step] = Tt_val, Ta_val, Tc_val, Tcem_val
        P_current, h_current = P_current + dP_calc, h_current + dh_calc
        dP_guess, dh_guess = dP_calc, dh_calc 
        
        TVD_current += dz * math.sin(theta_rad) 
        
        z_list_t.append(z_current + dz)
        TVD_list_t.append(TVD_current)
        theta_deg_list_t.append(theta_deg) 
        P_list_t.append(P_current / 1e6)
        h_list_t.append(h_current / 1000)
        x_list_t.append(x_avg)
        Ts_list_t.append(T_avg_K - 273.15)
        Tt_list_t.append(Tt_val - 273.15)
        Ta_list_t.append(Ta_val - 273.15)
        Tc_list_t.append(Tc_val - 273.15)
        Tcem_list_t.append(Tcem_val - 273.15)
        Tf_list_t.append(T_f_C) 

    Tt_prev, Ta_prev = Tt_curr, Ta_curr
    Tc_prev, Tcem_prev = Tc_curr, Tcem_curr
    
    is_target = any(abs(current_time_days - t_target) < 1e-5 for t_target in target_times)
    if is_target:
        print(f" -> 已记录: {current_time_days:.2f} 天")
        df_t = pd.DataFrame({
            "时间(天)": [round(current_time_days, 2)] * len(z_list_t),
            "测深MD(m)": z_list_t,
            "垂深TVD(m)": [round(v, 2) for v in TVD_list_t],
            "倾角(度)": [round(v, 2) for v in theta_deg_list_t], 
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
# 7. 导出数据
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

print(f"运算完成！代码已完美对齐接箍热桥修正与粗糙度模型。")
