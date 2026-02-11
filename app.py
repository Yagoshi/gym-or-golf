import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import json
import bisect
import subprocess
import tempfile
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import comb
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
from dataclasses import dataclass, field
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 4点追従用 PerformanceMetrics クラス改善版（完全版）
# 3点追従レベルの詳細な統計出力機能を追加
# ============================================================================

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクスを格納するデータクラス（4点追従用）"""
    
    # 横偏差
    lat_error_t1: List[float] = field(default_factory=list)
    lat_error_su1: List[float] = field(default_factory=list)
    lat_error_su2: List[float] = field(default_factory=list)
    lat_error_t2: List[float] = field(default_factory=list)
    
    # 方位偏差
    heading_error_t1: List[float] = field(default_factory=list)
    
    # 障害物クリアランス
    min_obstacle_dist: List[float] = field(default_factory=list)
    obstacle_clearance_t1: List[float] = field(default_factory=list)
    obstacle_clearance_su1: List[float] = field(default_factory=list)
    obstacle_clearance_su2: List[float] = field(default_factory=list)
    obstacle_clearance_t2: List[float] = field(default_factory=list)
    
    # 制御スムーズさ
    velocity_jerk: List[float] = field(default_factory=list)
    steering_jerk: List[float] = field(default_factory=list)
    
    # 計算パフォーマンス
    solve_times: List[float] = field(default_factory=list)
    solver_success: List[bool] = field(default_factory=list)
    
    # 進行状況
    arc_length_progress: List[float] = field(default_factory=list)
    
    # 速度追従
    velocity_actual: List[float] = field(default_factory=list)
    velocity_target: List[float] = field(default_factory=list)
    
    # 制御モード
    control_mode: List[str] = field(default_factory=list)
    
    def compute_statistics(self) -> Dict:
        """統計情報を計算"""
        stats = {}
        
        # 横偏差統計
        for name, data in [('lat_error_t1', self.lat_error_t1), 
                           ('lat_error_su1', self.lat_error_su1),
                           ('lat_error_su2', self.lat_error_su2),
                           ('lat_error_t2', self.lat_error_t2)]:
            if data:
                stats[name] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'max': float(np.max(np.abs(data))),
                    'rmse': float(np.sqrt(np.mean(np.array(data)**2)))
                }
        
        # 方位偏差統計
        if self.heading_error_t1:
            he_deg = np.rad2deg(self.heading_error_t1)
            stats['heading_error_t1_deg'] = {
                'mean': float(np.mean(he_deg)),
                'std': float(np.std(he_deg)),
                'max': float(np.max(np.abs(he_deg))),
                'rmse': float(np.sqrt(np.mean(he_deg**2)))
            }
        
        # 障害物クリアランス統計
        if self.min_obstacle_dist:
            stats['obstacle_clearance'] = {
                'min': float(np.min(self.min_obstacle_dist)),
                'mean': float(np.mean(self.min_obstacle_dist)),
                'violations': int(sum(1 for d in self.min_obstacle_dist if d < 0))
            }
        
        # 計算パフォーマンス統計
        if self.solve_times:
            stats['computation'] = {
                'mean_ms': float(np.mean(self.solve_times)),
                'max_ms': float(np.max(self.solve_times)),
                'std_ms': float(np.std(self.solve_times)),
                'total_s': float(np.sum(self.solve_times) / 1000),
                'success_rate': float(sum(self.solver_success) / len(self.solver_success) * 100) if self.solver_success else 0
            }
        
        # 速度追従統計
        if self.velocity_actual and self.velocity_target:
            vel_error = np.array(self.velocity_actual) - np.array(self.velocity_target)
            stats['velocity_tracking'] = {
                'mean_error': float(np.mean(vel_error)),
                'rmse': float(np.sqrt(np.mean(vel_error**2))),
                'max_error': float(np.max(np.abs(vel_error)))
            }
        
        # スムーズさ統計
        if self.velocity_jerk:
            stats['smoothness'] = {
                'velocity_jerk_mean': float(np.mean(np.abs(self.velocity_jerk))),
                'velocity_jerk_max': float(np.max(np.abs(self.velocity_jerk))),
                'steering_jerk_mean': float(np.mean(np.abs(self.steering_jerk))) if self.steering_jerk else 0,
                'steering_jerk_max': float(np.max(np.abs(self.steering_jerk))) if self.steering_jerk else 0
            }
        
        return stats
    
    def print_summary(self, dt: float):
        """詳細なサマリーをコンソールに出力"""
        stats = self.compute_statistics()
        
        print("\n" + "=" * 80)
        print("  PERFORMANCE METRICS SUMMARY (4-Point Tracking)")
        print("=" * 80)
        
        print("\n--- TRACKING PERFORMANCE ---")
        if 'lat_error_t1' in stats:
            s = stats['lat_error_t1']
            print(f"  Tractor 1 Rear:  RMSE={s['rmse']:.4f}m, Max={s['max']:.4f}m, Std={s['std']:.4f}m")
        if 'lat_error_su1' in stats:
            s = stats['lat_error_su1']
            print(f"  Steering Unit 1: RMSE={s['rmse']:.4f}m, Max={s['max']:.4f}m, Std={s['std']:.4f}m")
        if 'lat_error_su2' in stats:
            s = stats['lat_error_su2']
            print(f"  Steering Unit 2: RMSE={s['rmse']:.4f}m, Max={s['max']:.4f}m, Std={s['std']:.4f}m")
        if 'lat_error_t2' in stats:
            s = stats['lat_error_t2']
            print(f"  Tractor 2 Rear:  RMSE={s['rmse']:.4f}m, Max={s['max']:.4f}m, Std={s['std']:.4f}m")
        
        if 'heading_error_t1_deg' in stats:
            s = stats['heading_error_t1_deg']
            print(f"  Heading Error (T1): RMSE={s['rmse']:.2f}deg, Max={s['max']:.2f}deg")
        
        print("\n--- OBSTACLE AVOIDANCE ---")
        if 'obstacle_clearance' in stats:
            s = stats['obstacle_clearance']
            status = "SAFE" if s['violations'] == 0 else f"{s['violations']} VIOLATIONS"
            print(f"  Min Clearance: {s['min']:.3f}m, Mean: {s['mean']:.3f}m, Status: {status}")
        
        print("\n--- COMPUTATION PERFORMANCE ---")
        if 'computation' in stats:
            s = stats['computation']
            print(f"  Solve Time: Mean={s['mean_ms']:.2f}ms, Max={s['max_ms']:.2f}ms, Total={s['total_s']:.2f}s")
            print(f"  Success Rate: {s['success_rate']:.1f}%")
            
            # リアルタイム性能の評価
            if len(self.solve_times) > 0 and s['total_s'] > 0:
                rt_factor = (len(self.solve_times) * dt) / s['total_s']
                print(f"  Real-time Factor: {rt_factor:.2f}x (>1.0 = real-time capable)")
        
        print("\n--- VELOCITY TRACKING ---")
        if 'velocity_tracking' in stats:
            s = stats['velocity_tracking']
            print(f"  Velocity Error: RMSE={s['rmse']:.4f}m/s, Max={s['max_error']:.4f}m/s")
        
        print("\n--- CONTROL SMOOTHNESS ---")
        if 'smoothness' in stats:
            s = stats['smoothness']
            print(f"  Velocity Jerk: Mean={s['velocity_jerk_mean']:.4f}, Max={s['velocity_jerk_max']:.4f} m/s^3")
            print(f"  Steering Jerk: Mean={s['steering_jerk_mean']:.4f}, Max={s['steering_jerk_max']:.4f} rad/s^3")
        
        print("=" * 80)
        
        return stats
    
# ==============================================================================
# PART 1: Bezier Editor UI (Unchanged)
# ==============================================================================
# (省略: 元のコードと同じ内容です)
class BezierObstacleEditor:
    def __init__(self, control_points, obstacles, on_export_callback=None):
        self.control_points = np.array(control_points, dtype=float)
        self.obstacles = obstacles
        self.safety_margin = 1.2
        self.default_obstacle_radius = 2.0
        self.on_export_callback = on_export_callback
        self.selected_point_index = None
        self.selected_obstacle_index = None
        self.dragging_type = None
        self.is_dragging = False
        self.config = None

        self.fig = plt.figure(figsize=(14, 9))
        self.ax = self.fig.add_axes([0.1, 0.25, 0.85, 0.7])
        self.ax.set_title("Bezier Editor | Left:drag | Right:add/remove obstacle | Shift+Left:add point")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-5, 35)
        self.ax.set_ylim(-5, 35)
        self.ax.set_aspect('equal')

        self.polygon_line, = self.ax.plot([], [], 'k--', alpha=0.5)
        self.points_scatter, = self.ax.plot([], [], 'ro', markersize=10, picker=5)
        self.curve_line, = self.ax.plot([], [], 'b-', linewidth=2)
        self.obstacle_patches = []
        self.margin_patches = []

        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, verticalalignment='top', fontsize=9,
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax_margin = self.fig.add_axes([0.15, 0.15, 0.6, 0.03])
        self.slider_margin = Slider(ax_margin, 'Safety Margin', 0.0, 5.0, valinit=self.safety_margin, valstep=0.1)
        self.slider_margin.on_changed(self.update_safety_margin)
        
        ax_obstacle = self.fig.add_axes([0.15, 0.09, 0.6, 0.03])
        self.slider_obstacle = Slider(ax_obstacle, 'Obstacle Radius', 0.5, 5.0, valinit=self.default_obstacle_radius, valstep=0.1)
        self.slider_obstacle.on_changed(self.update_obstacle_radius)

        ax_export = self.fig.add_axes([0.78, 0.14, 0.15, 0.05])
        self.btn_export = Button(ax_export, 'Save & Run MPC')
        self.btn_export.on_clicked(self.export_configuration)

        self.update_plot()
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def compute_bezier(self):
        n = len(self.control_points) - 1
        t = np.linspace(0, 1, 100)
        curve_x, curve_y = np.zeros_like(t), np.zeros_like(t)
        for i in range(n + 1):
            b = comb(n, i) * ((1 - t)**(n - i)) * (t**i)
            curve_x += self.control_points[i, 0] * b
            curve_y += self.control_points[i, 1] * b
        return curve_x, curve_y

    def check_collision(self):
        cx, cy = self.compute_bezier()
        curve_points = np.column_stack([cx, cy])
        for obs in self.obstacles:
            center = np.array(obs['center'])
            radius_with_margin = obs['radius'] + self.safety_margin
            if np.any(np.linalg.norm(curve_points - center, axis=1) < radius_with_margin):
                return True
        return False

    def update_plot(self):
        self.polygon_line.set_data(self.control_points[:, 0], self.control_points[:, 1])
        self.points_scatter.set_data(self.control_points[:, 0], self.control_points[:, 1])
        cx, cy = self.compute_bezier()
        collision = self.check_collision()
        self.curve_line.set_data(cx, cy)
        self.curve_line.set_color('red' if collision else 'blue')
        for patch in self.obstacle_patches + self.margin_patches:
            patch.remove()
        self.obstacle_patches.clear()
        self.margin_patches.clear()
        for i, obs in enumerate(self.obstacles):
            center, radius = obs['center'], obs['radius']
            margin_circle = Circle(center, radius + self.safety_margin, fill=False, edgecolor='orange', linestyle='--', linewidth=1.5, alpha=0.6)
            self.ax.add_patch(margin_circle)
            self.margin_patches.append(margin_circle)
            facecolor = 'lightblue' if i == self.selected_obstacle_index else 'gray'
            edgecolor = 'blue' if i == self.selected_obstacle_index else 'black'
            obstacle_circle = Circle(center, radius, fill=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=2, alpha=0.7)
            self.ax.add_patch(obstacle_circle)
            self.obstacle_patches.append(obstacle_circle)
        status = "COLLISION!" if collision else "Safe"
        self.info_text.set_text(f"Points: {len(self.control_points)}\nObstacles: {len(self.obstacles)}\nMargin: {self.safety_margin:.1f}\nStatus: {status}")
        self.fig.canvas.draw()

    def update_safety_margin(self, val):
        self.safety_margin = val
        self.update_plot()

    def update_obstacle_radius(self, val):
        self.default_obstacle_radius = val
        if self.selected_obstacle_index is not None:
            self.obstacles[self.selected_obstacle_index]['radius'] = val
            self.update_plot()

    def point_to_segment_distance(self, point, seg_start, seg_end):
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0:
            return np.linalg.norm(point - seg_start)
        t = max(0, min(1, np.dot(point - seg_start, seg_vec) / seg_len_sq))
        return np.linalg.norm(point - (seg_start + t * seg_vec))

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        mouse = np.array([event.xdata, event.ydata])
        is_shift = event.key == 'shift' or (hasattr(event, 'key') and event.key and 'shift' in str(event.key).lower())
        
        if event.button == 2 or (event.button == 1 and is_shift):
            min_dist, insert_idx = float('inf'), len(self.control_points)
            for i in range(len(self.control_points) - 1):
                dist = self.point_to_segment_distance(mouse, self.control_points[i], self.control_points[i + 1])
                if dist < min_dist:
                    min_dist, insert_idx = dist, i + 1
            self.control_points = np.insert(self.control_points, insert_idx, [[round(event.xdata), round(event.ydata)]], axis=0)
            self.update_plot()
            return
        if event.button == 3 and is_shift:
            if len(self.control_points) <= 3:
                return
            dists = np.linalg.norm(self.control_points - mouse, axis=1)
            if np.min(dists) < 1.0:
                self.control_points = np.delete(self.control_points, np.argmin(dists), axis=0)
                self.update_plot()
            return
        if event.button == 3 and not is_shift:
            for i, obs in enumerate(self.obstacles):
                if np.linalg.norm(mouse - np.array(obs['center'])) < obs['radius']:
                    self.obstacles.pop(i)
                    self.update_plot()
                    return
            self.obstacles.append({'center': [round(event.xdata), round(event.ydata)], 'radius': self.default_obstacle_radius})
            self.update_plot()
            return
        if event.button == 1 and not is_shift:
            for i, obs in enumerate(self.obstacles):
                if np.linalg.norm(mouse - np.array(obs['center'])) < obs['radius'] + self.safety_margin:
                    self.selected_obstacle_index, self.selected_point_index = i, None
                    self.dragging_type, self.is_dragging = 'obstacle', True
                    self.slider_obstacle.set_val(obs['radius'])
                    self.update_plot()
                    return
            dists = np.linalg.norm(self.control_points - mouse, axis=1)
            if np.min(dists) < 1.0:
                self.selected_point_index, self.selected_obstacle_index = np.argmin(dists), None
                self.dragging_type, self.is_dragging = 'control_point', True
                self.update_plot()
                return
            self.selected_obstacle_index = self.selected_point_index = None
            self.update_plot()

    def on_motion(self, event):
        if event.inaxes != self.ax or not self.is_dragging:
            return
        new_x, new_y = round(event.xdata), round(event.ydata)
        if self.dragging_type == 'control_point' and self.selected_point_index is not None:
            self.control_points[self.selected_point_index] = [new_x, new_y]
            self.update_plot()
        elif self.dragging_type == 'obstacle' and self.selected_obstacle_index is not None:
            self.obstacles[self.selected_obstacle_index]['center'] = [new_x, new_y]
            self.update_plot()

    def on_release(self, event):
        self.is_dragging, self.dragging_type = False, None

    def on_key_press(self, event):
        if event.key == 'escape':
            self.selected_obstacle_index = self.selected_point_index = None
            self.update_plot()

    def export_configuration(self, event):
        self.config = {'control_points': self.control_points.tolist(), 'obstacles': self.obstacles, 'safety_margin': self.safety_margin}
        print(f"\n=== Configuration Saved ===\nPoints: {len(self.config['control_points'])}, Obstacles: {len(self.config['obstacles'])}, Margin: {self.safety_margin:.2f}\n")
        plt.close(self.fig)
        if self.on_export_callback:
            self.on_export_callback(self.config)

    def show(self):
        plt.show()
        return self.config


# ==============================================================================
# PART 2: MPC Class (Modified for 4-point tracking)
# ==============================================================================

class MultiArticulatedVehicleMPC:
    def __init__(self):
        self.L1, self.L2, self.L3, self.L4 = 1.0, 1.0, 1.0, 3.0
        self.L5, self.L6, self.L7 = 1.0, 1.0, 1.0
        self.R1, self.R2 = 0.5, 0.5
        self.dt = 0.1
        self.N = 20
        self.nx, self.nu = 11, 5
        self.phi_max, self.v1_max = np.deg2rad(60), 3.0
        self.vehicle_safety_radius = 1.2
        # ウォームスタート用ラグランジュ乗数
        self._lam_g0 = None
        self._lam_x0 = None
        self.setup_mpc()
    
    def compute_vehicle_positions_casadi(self, X):
        """CasADi変数を使用した座標計算（元と完全同一のロジック）"""
        x1, y1, theta_a1, theta_a2, theta_a4, theta_a6, theta_a7 = X[0], X[1], X[2], X[4], X[6], X[8], X[9]
        t1_rear = ca.vertcat(x1, y1)
        hitch1 = ca.vertcat(x1 - self.R1 * ca.cos(theta_a1), y1 - self.R1 * ca.sin(theta_a1))
        steer1 = ca.vertcat(hitch1[0] - (self.L2 + self.L3) * ca.cos(theta_a2), hitch1[1] - (self.L2 + self.L3) * ca.sin(theta_a2))
        conn1 = ca.vertcat(steer1[0] + self.L3 * ca.cos(theta_a2), steer1[1] + self.L3 * ca.sin(theta_a2))
        conn2 = ca.vertcat(conn1[0] - self.L4 * ca.cos(theta_a4), conn1[1] - self.L4 * ca.sin(theta_a4))
        steer2 = ca.vertcat(conn2[0] + self.L5 * ca.cos(theta_a6), conn2[1] + self.L5 * ca.sin(theta_a6))
        hitch2 = ca.vertcat(steer2[0] - (self.L6 + self.L5) * ca.cos(theta_a6), steer2[1] - (self.L6 + self.L5) * ca.sin(theta_a6))
        t2_rear = ca.vertcat(hitch2[0] - self.R2 * ca.cos(theta_a7), hitch2[1] - self.R2 * ca.sin(theta_a7))
        return {'tractor1_rear': t1_rear, 'steer1': steer1, 'steer2': steer2, 'tractor2_rear': t2_rear}
    
    def compute_g1_elements(self, X):
        theta_a1, phi_a1, theta_a2, theta_a3 = X[2], X[3], X[4], X[5]
        theta_a4, theta_a5, theta_a6, theta_a7, phi_a2 = X[6], X[7], X[8], X[9], X[10]
        L_23, L_56, eps = self.L2 + self.L3, self.L5 + self.L6, 1e-8
        theta_a12, theta_a13, theta_a14 = theta_a1 - theta_a2, theta_a1 - theta_a3, theta_a1 - theta_a4
        theta_a23, theta_a24 = theta_a2 - theta_a3, theta_a2 - theta_a4
        theta_a46, theta_a47 = theta_a4 - theta_a6, theta_a4 - theta_a7
        theta_a56, theta_a57, theta_a67 = theta_a5 - theta_a6, theta_a5 - theta_a7, theta_a6 - theta_a7
        G13 = ca.tan(phi_a1) / self.L1
        G15 = (ca.sin(theta_a13) - self.R1 * G13 * ca.cos(theta_a13)) / (L_23 * ca.cos(theta_a23) + eps)
        l1 = ca.cos(theta_a12) * (self.L1 * L_23 * ca.cos(theta_a24) + ca.sin(theta_a24) * (self.L3 * self.R1 * ca.tan(phi_a1) + self.L1 * self.L2 * ca.tan(theta_a23)))
        l2 = ca.sin(theta_a12) * (L_23 * self.R1 * ca.cos(theta_a24) * ca.tan(phi_a1) + ca.sin(theta_a24) * (-self.L1 * self.L3 + self.L2 * self.R1 * ca.tan(phi_a1) * ca.tan(theta_a23)))
        l3 = self.L7 * ca.cos(theta_a67) - self.R2 * ca.sin(theta_a67) * ca.tan(phi_a2)
        l4 = ca.cos(theta_a67) * (self.L5 * self.R2 * ca.tan(phi_a2) - self.L6 * self.L7 * ca.tan(theta_a56)) + ca.sin(theta_a67) * (self.L5 * self.L7 + self.L6 * self.R2 * ca.tan(phi_a2) * ca.tan(theta_a56))
        denom = self.L1 * L_23 * (L_56 * ca.cos(theta_a46) * l3 - ca.sin(theta_a46) * l4)
        V_rear = (L_56 * self.L7 * (l1 + l2)) / (denom + eps)
        G_1_10 = -V_rear * ca.tan(phi_a2) / self.L7
        G19 = (V_rear * ca.sin(theta_a57) - self.R2 * G_1_10 * ca.cos(theta_a57)) / (L_56 * ca.cos(theta_a56) + eps)
        Lambda = self.R1 * G13 * ca.cos(theta_a14) + self.L2 * G15 * ca.cos(theta_a24) + self.L6 * G19 * ca.cos(theta_a46) + self.R2 * G_1_10 * ca.cos(theta_a47) - ca.sin(theta_a14) - V_rear * ca.sin(theta_a47)
        G17 = -Lambda / self.L4
        return G13, G15, G17, G19, G_1_10, V_rear
        
    def vehicle_dynamics(self, X, U):
        v1, v2, v3, v4, v5, theta_a1 = U[0], U[1], U[2], U[3], U[4], X[2]
        G13, G15, G17, G19, G_1_10, _ = self.compute_g1_elements(X)
        return ca.vertcat(ca.cos(theta_a1)*v1, ca.sin(theta_a1)*v1, G13*v1, v2, G15*v1, v3, G17*v1, v4, G19*v1, G_1_10*v1, v5)
    
    def setup_mpc(self):
        """
        高速版MPC: casadi.nlpsol + CodeGen
        
        決定変数 w = [X_0, U_0, X_1, U_1, ..., X_{N-1}, U_{N-1}, X_N]
        パラメータ p = [x0, t1_ref_flat, su1_ref_flat, su2_ref_flat, t2_ref_flat, v_ref, obs, weights]
        """
        nx, nu, N, dt = self.nx, self.nu, self.N, self.dt
        
        print("  [MPC] Building NLP with nlpsol (high-performance mode)...")
        t_build_start = time.time()
        
        # 離散化関数 F (シミュレーション用)
        X_sym = ca.MX.sym('X', nx)
        U_sym = ca.MX.sym('U', nu)
        self.F = ca.Function('F', [X_sym, U_sym], [X_sym + dt * self.vehicle_dynamics(X_sym, U_sym)])
        
        # 決定変数
        n_w = (N + 1) * nx + N * nu
        w = ca.MX.sym('w', n_w)
        
        # パラメータ
        n_ref = 2 * (N + 1)
        n_p = nx + 4 * n_ref + N + 3 + 10
        p = ca.MX.sym('p', n_p)
        
        # パラメータ切り出し
        idx = 0
        p_x0 = p[idx:idx+nx]; idx += nx
        p_t1 = p[idx:idx+n_ref]; idx += n_ref
        p_su1 = p[idx:idx+n_ref]; idx += n_ref
        p_su2 = p[idx:idx+n_ref]; idx += n_ref
        p_t2 = p[idx:idx+n_ref]; idx += n_ref
        p_vr = p[idx:idx+N]; idx += N
        p_obs = p[idx:idx+3]; idx += 3
        p_w = p[idx:idx+10]; idx += 10
        # p_w: [0]Q_pos_t1 [1]Q_pos_su1 [2]Q_pos_su2 [3]Q_pos_t2
        #      [4]Q_angle [5]R [6]Q_vel [7]Q_phi_rate [8]Q_phi_mag [9]R_rate
        
        # 変数インデックスヘルパー
        def get_X(k):
            s = k * (nx + nu) if k < N else N * (nx + nu)
            return w[s:s+nx]
        def get_U(k):
            s = k * (nx + nu) + nx
            return w[s:s+nu]
        
        # 制約定数
        max_art = np.deg2rad(80)
        max_sr = np.deg2rad(60)
        max_dv = 0.2
        max_ds = np.deg2rad(10)
        
        # コスト関数 & 制約
        J = 0
        g_list, lbg_list, ubg_list = [], [], []
        
        # 初期条件
        g_list.append(get_X(0) - p_x0)
        lbg_list += [0.0] * nx
        ubg_list += [0.0] * nx
        
        for k in range(N + 1):
            Xk = get_X(k)
            pos = self.compute_vehicle_positions_casadi(Xk)
            
            # 障害物制約 (4点)
            safe_sq = (p_obs[2] + self.vehicle_safety_radius)**2
            for key in ['tractor1_rear', 'steer1', 'steer2', 'tractor2_rear']:
                pt = pos[key]
                d_sq = (pt[0] - p_obs[0])**2 + (pt[1] - p_obs[1])**2
                g_list.append(d_sq - safe_sq)
                lbg_list.append(0.0)
                ubg_list.append(1e20)
            
            if k < N:
                Uk = get_U(k)
                Xnext = get_X(k + 1)
                
                # ダイナミクス制約
                Xpred = Xk + dt * self.vehicle_dynamics(Xk, Uk)
                g_list.append(Xnext - Xpred)
                lbg_list += [0.0] * nx
                ubg_list += [0.0] * nx
                
                # コスト: 4点追従
                t1r = p_t1[2*k:2*k+2]
                su1r = p_su1[2*k:2*k+2]
                su2r = p_su2[2*k:2*k+2]
                t2r = p_t2[2*k:2*k+2]
                
                J += p_w[0] * ca.dot(pos['tractor1_rear'] - t1r, pos['tractor1_rear'] - t1r)
                J += p_w[1] * ca.dot(pos['steer1'] - su1r, pos['steer1'] - su1r)
                J += p_w[2] * ca.dot(pos['steer2'] - su2r, pos['steer2'] - su2r)
                J += p_w[3] * ca.dot(pos['tractor2_rear'] - t2r, pos['tractor2_rear'] - t2r)
                
                J += p_w[4] * (Xk[2]**2 + Xk[4]**2 + Xk[6]**2 + Xk[9]**2)
                J += p_w[8] * (Xk[3]**2 + Xk[10]**2)
                J += p_w[5] * (Uk[1]**2 + Uk[2]**2 + Uk[3]**2 + Uk[4]**2)
                J += p_w[6] * (Uk[0] - p_vr[k])**2
                J += p_w[7] * (Uk[1]**2 + Uk[4]**2)
                
                if k > 0:
                    du = Uk - get_U(k - 1)
                    J += p_w[9] * ca.dot(du, du)
                    g_list.append(du[0]); lbg_list.append(-max_dv); ubg_list.append(max_dv)
                    g_list.append(du[1]); lbg_list.append(-max_ds); ubg_list.append(max_ds)
                    g_list.append(du[4]); lbg_list.append(-max_ds); ubg_list.append(max_ds)
                
                # 関節角差制約
                g_list.append(Xk[4] - Xk[5]); lbg_list.append(-max_art); ubg_list.append(max_art)
                g_list.append(Xk[7] - Xk[8]); lbg_list.append(-max_art); ubg_list.append(max_art)
            else:
                # 終端コスト (5倍)
                t1r = p_t1[2*k:2*k+2]
                su1r = p_su1[2*k:2*k+2]
                su2r = p_su2[2*k:2*k+2]
                t2r = p_t2[2*k:2*k+2]
                J += 5.0 * p_w[0] * ca.dot(pos['tractor1_rear'] - t1r, pos['tractor1_rear'] - t1r)
                J += 5.0 * p_w[1] * ca.dot(pos['steer1'] - su1r, pos['steer1'] - su1r)
                J += 5.0 * p_w[2] * ca.dot(pos['steer2'] - su2r, pos['steer2'] - su2r)
                J += 5.0 * p_w[3] * ca.dot(pos['tractor2_rear'] - t2r, pos['tractor2_rear'] - t2r)
        
        g = ca.vertcat(*g_list)
        self.lbg = np.array(lbg_list)
        self.ubg = np.array(ubg_list)
        
        # 変数の上下限
        lbw = np.full(n_w, -1e20)
        ubw = np.full(n_w, 1e20)
        for k in range(N):
            xs = k * (nx + nu)
            lbw[xs+3] = -self.phi_max; ubw[xs+3] = self.phi_max
            lbw[xs+10] = -self.phi_max; ubw[xs+10] = self.phi_max
            us = xs + nx
            lbw[us+0] = 0.2; ubw[us+0] = self.v1_max
            lbw[us+2] = -max_sr; ubw[us+2] = max_sr
            lbw[us+3] = -max_sr; ubw[us+3] = max_sr
        xs_N = N * (nx + nu)
        lbw[xs_N+3] = -self.phi_max; ubw[xs_N+3] = self.phi_max
        lbw[xs_N+10] = -self.phi_max; ubw[xs_N+10] = self.phi_max
        
        self.lbw = lbw; self.ubw = ubw
        self.n_w = n_w; self.n_p = n_p
        self._nx = nx; self._nu = nu; self._N = N
        
        nlp = {'x': w, 'f': J, 'g': g, 'p': p}
        opts = {
            'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.max_iter': 100, 'ipopt.tol': 1e-2, 'ipopt.acceptable_tol': 1e-3, 'ipopt.acceptable_iter': 3,
            'ipopt.warm_start_init_point': 'yes', 'ipopt.warm_start_bound_push': 1e-6, 'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.mu_strategy': 'adaptive', 'ipopt.mu_init': 1e-1, 'ipopt.linear_solver': 'mumps', 'ipopt.sb': 'yes',
        }
        
        # CodeGen を試みる
        self.codegen_used = False
        try:
            print("  [MPC] Attempting CodeGen compilation...")
            cg_dir = tempfile.mkdtemp(prefix='mpc_codegen_')
            solver_plain = ca.nlpsol('solver', 'ipopt', nlp, opts)
            c_file = 'mpc_nlp.c'
            so_file = os.path.join(cg_dir, 'mpc_nlp.so')
            solver_plain.generate_dependencies(c_file, {'with_header': False})
            c_path = os.path.join(os.getcwd(), c_file) if os.path.exists(c_file) else c_file
            ret = subprocess.run(['gcc', '-O3', '-march=native', '-shared', '-fPIC', c_path, '-o', so_file],
                                 capture_output=True, text=True, timeout=60)
            if ret.returncode == 0 and os.path.exists(so_file):
                self.solver = ca.nlpsol('solver', 'ipopt', so_file, opts)
                self.codegen_used = True
                print(f"  [MPC] CodeGen SUCCESS")
                if os.path.exists(c_path): os.remove(c_path)
            else:
                print(f"  [MPC] CodeGen compile failed, using standard nlpsol")
                self.solver = solver_plain
                if os.path.exists(c_path): os.remove(c_path)
        except Exception as e:
            print(f"  [MPC] CodeGen failed ({e}), using standard nlpsol")
            self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        t_build = time.time() - t_build_start
        mode = "CodeGen" if self.codegen_used else "nlpsol"
        print(f"  [MPC] NLP built in {t_build:.2f}s (mode: {mode}, vars={n_w}, cons={len(lbg_list)}, params={n_p})")
    
    def _pack_w0(self, X_guess, U_guess):
        nx, nu, N = self._nx, self._nu, self._N
        w0 = np.zeros(self.n_w)
        for k in range(N):
            s = k * (nx + nu)
            w0[s:s+nx] = X_guess[:, k]
            w0[s+nx:s+nx+nu] = U_guess[:, k]
        w0[N*(nx+nu):N*(nx+nu)+nx] = X_guess[:, N]
        return w0
    
    def _pack_p(self, x0, t1r, su1r, su2r, t2r, vr, obs, wts):
        w_arr = np.array([wts['Q_pos_t1'], wts['Q_pos_su1'], wts['Q_pos_su2'], wts['Q_pos_t2'],
                          wts['Q_angle'], wts['R'], wts['Q_vel'], wts['Q_phi_rate'], wts['Q_phi_mag'], wts['R_rate']])
        return np.concatenate([x0, t1r.T.flatten(), su1r.T.flatten(), su2r.T.flatten(), t2r.T.flatten(),
                               vr, np.array(obs[:3], dtype=float), w_arr])
    
    def _unpack_w(self, w_opt):
        nx, nu, N = self._nx, self._nu, self._N
        w = np.array(w_opt).flatten()
        X = np.zeros((nx, N + 1)); U = np.zeros((nu, N))
        for k in range(N):
            s = k * (nx + nu)
            X[:, k] = w[s:s+nx]; U[:, k] = w[s+nx:s+nx+nu]
        X[:, N] = w[N*(nx+nu):N*(nx+nu)+nx]
        return X, U
    
    def solve(self, x0, t1_ref_val, su1_ref_val, su2_ref_val, t2_ref_val, obs_params, v_ref_array, weights, X_guess=None, U_guess=None):
        try:
            p_val = self._pack_p(x0, t1_ref_val, su1_ref_val, su2_ref_val, t2_ref_val, v_ref_array, obs_params, weights)
            w0 = self._pack_w0(X_guess, U_guess) if (X_guess is not None and U_guess is not None) else np.zeros(self.n_w)
            args = {'x0': w0, 'lbx': self.lbw, 'ubx': self.ubw, 'lbg': self.lbg, 'ubg': self.ubg, 'p': p_val}
            if self._lam_g0 is not None: args['lam_g0'] = self._lam_g0
            if self._lam_x0 is not None: args['lam_x0'] = self._lam_x0
            
            t0 = time.time()
            sol = self.solver(**args)
            t_solve = (time.time() - t0) * 1000
            
            if self.solver.stats()['success']:
                self._lam_g0 = np.array(sol['lam_g']).flatten()
                self._lam_x0 = np.array(sol['lam_x']).flatten()
                X_plan, U_plan = self._unpack_w(sol['x'])
                return U_plan[:, 0], X_plan, U_plan, t_solve, True
            else:
                return np.zeros(self.nu), None, None, t_solve, False
        except:
            return np.zeros(self.nu), None, None, 0.0, False


# ==============================================================================
# PART 3: Utility Functions
# ==============================================================================

def compute_vehicle_positions_numpy(x, L1, L2, L3, L4, L5, L6, L7, R1, R2):
    """
    NumPy配列用の座標計算関数
    ターゲットとなる4点 (t1_rear, steer1, steer2, t2_rear) を返すように変更
    """
    x1, y1 = x[0], x[1]
    theta_a1 = x[2]
    theta_a2 = x[4]
    theta_a4 = x[6]
    theta_a6 = x[8]
    theta_a7 = x[9]
    
    # 1. Tractor 1 Rear (State variables x1, y1 are already rear axle center)
    t1_rear = np.array([x1, y1])
    
    # Hitch 1
    hitch1 = np.array([x1 - R1 * np.cos(theta_a1), y1 - R1 * np.sin(theta_a1)])
    
    # 2. Steering Unit 1
    # Defined relative to Hitch1 in the model
    steer1 = np.array([
        hitch1[0] - (L2 + L3) * np.cos(theta_a2),
        hitch1[1] - (L2 + L3) * np.sin(theta_a2)
    ])
    
    # Connection 1
    conn1 = np.array([steer1[0] + L3 * np.cos(theta_a2), steer1[1] + L3 * np.sin(theta_a2)])
    
    # Connection 2
    conn2 = np.array([conn1[0] - L4 * np.cos(theta_a4), conn1[1] - L4 * np.sin(theta_a4)])
    
    # 3. Steering Unit 2
    steer2 = np.array([
        conn2[0] + L5 * np.cos(theta_a6),
        conn2[1] + L5 * np.sin(theta_a6)
    ])
    
    # Hitch 2
    hitch2 = np.array([steer2[0] - (L6 + L5) * np.cos(theta_a6), steer2[1] - (L6 + L5) * np.sin(theta_a6)])
    
    # 4. Tractor 2 Rear
    t2_rear = np.array([
        hitch2[0] - R2 * np.cos(theta_a7),
        hitch2[1] - R2 * np.sin(theta_a7)
    ])
    
    return t1_rear, steer1, steer2, t2_rear

def get_bezier_functions(points):
    """高速ベジェ曲線関数（二項係数を事前計算 + ベクトル化）"""
    points = np.array(points)
    n = len(points) - 1
    binom_coeffs = np.array([comb(n, i, exact=True) for i in range(n + 1)], dtype=float)
    if n >= 1:
        deriv_coeffs = np.array([comb(n-1, i, exact=True) for i in range(n)], dtype=float)
        deriv_points = n * np.diff(points, axis=0)
    def B(t):
        t = np.clip(t, 0.0, 1.0)
        i_arr = np.arange(n + 1)
        basis = binom_coeffs * ((1 - t) ** (n - i_arr)) * (t ** i_arr)
        return basis @ points
    def B_prime(t):
        t = np.clip(t, 0.0, 1.0)
        if n < 1:
            return np.zeros(2)
        i_arr = np.arange(n)
        basis = deriv_coeffs * ((1 - t) ** (n - 1 - i_arr)) * (t ** i_arr)
        return basis @ deriv_points
    def B_batch(t_array):
        """バッチ評価"""
        t = np.clip(np.asarray(t_array), 0.0, 1.0)[:, None]
        i_arr = np.arange(n + 1)
        basis = binom_coeffs * ((1 - t) ** (n - i_arr)) * (t ** i_arr)
        return basis @ points
    return B, B_prime, B_batch

def calculate_target_velocity(t_current, s_current, total_length, v_max=3.0, accel_time=5.0, decel_accel=0.2):
    v_accel = v_max * min(1, t_current / accel_time)
    v_decel = np.sqrt(2 * decel_accel * max(0, total_length - s_current))
    v_target = max(0.05 if total_length - s_current > 0.1 else 0, min(v_accel, v_decel))
    return v_target

def initialize_vehicle_along_bezier(B, B_prime, s_to_t, s_init, mpc, lat_error=0.0, head_error_deg=0.0):
    L1, L2, L3, L4, L5, L6, L7 = mpc.L1, mpc.L2, mpc.L3, mpc.L4, mpc.L5, mpc.L6, mpc.L7
    R1, R2 = mpc.R1, mpc.R2
    t_t1 = s_to_t(s_init)
    pos_t1, tan_t1 = B(t_t1), B_prime(t_t1)
    theta_base = np.arctan2(tan_t1[1], tan_t1[0])
    
    # 幾何学的な遅れ距離の計算
    dist_to_steer1 = R1 + L2 + L3
    dist_to_cargo_conn1 = dist_to_steer1 - L3 # connection point
    dist_to_cargo_conn2 = dist_to_cargo_conn1 + L4
    dist_to_steer2 = dist_to_cargo_conn2 - L5
    dist_to_t2 = dist_to_steer2 + L6 + L5 + R2

    theta_a2 = np.arctan2(B_prime(s_to_t(max(0, s_init - dist_to_steer1)))[1], B_prime(s_to_t(max(0, s_init - dist_to_steer1)))[0])
    theta_a4 = np.arctan2(B_prime(s_to_t(max(0, s_init - (dist_to_cargo_conn1 + L4/2))))[1], B_prime(s_to_t(max(0, s_init - (dist_to_cargo_conn1 + L4/2))))[0])
    theta_a6 = np.arctan2(B_prime(s_to_t(max(0, s_init - dist_to_steer2)))[1], B_prime(s_to_t(max(0, s_init - dist_to_steer2)))[0])
    theta_a7 = np.arctan2(B_prime(s_to_t(max(0, s_init - dist_to_t2)))[1], B_prime(s_to_t(max(0, s_init - dist_to_t2)))[0])
    
    head_err = np.deg2rad(head_error_deg)
    norm = np.linalg.norm(tan_t1)
    nx, ny = -tan_t1[1] / norm, tan_t1[0] / norm
    return np.array([pos_t1[0] + nx * lat_error, pos_t1[1] + ny * lat_error, theta_base + head_err, 0.0, 
                     theta_a2 + head_err, 0.0, theta_a4 + head_err, 0.0, theta_a6 + head_err, theta_a7 + head_err, 0.0])

def find_closest_point(pos, B, search_range=(0, 1.0), t_hint=None):
    """最近接点探索（t_hintがあれば探索範囲を限定して高速化）"""
    pos = np.array(pos)
    if t_hint is not None:
        sr = (max(0.0, t_hint - 0.15), min(1.0, t_hint + 0.15))
    else:
        sr = search_range
    result = minimize_scalar(lambda t: np.sum((B(t) - pos)**2), bounds=sr, method='bounded')
    return result.x if result.success else sr[0]

def create_arc_length_interpolator(B_prime):
    """弧長パラメータ化テーブル（シンプソン法で高速化）"""
    t_samples = np.linspace(0, 1, 200)
    s_samples = np.zeros(200)
    for i in range(1, 200):
        t0, t1 = t_samples[i-1], t_samples[i]
        tm = (t0 + t1) / 2.0
        f0 = np.linalg.norm(B_prime(t0))
        fm = np.linalg.norm(B_prime(tm))
        f1 = np.linalg.norm(B_prime(t1))
        s_samples[i] = s_samples[i-1] + (t1 - t0) / 6.0 * (f0 + 4*fm + f1)
    # np.interp を直接使うラッパー (interp1dよりオーバーヘッドが少ない)
    def s_to_t(s): return float(np.interp(s, s_samples, t_samples))
    def t_to_s(t): return float(np.interp(t, t_samples, s_samples))
    def s_to_t_batch(s_arr): return np.interp(s_arr, s_samples, t_samples)
    return s_to_t, t_to_s, s_samples[-1], s_to_t_batch, s_samples, t_samples

def compute_lateral_error(pos, B, B_prime):
    t_closest = find_closest_point(pos, B)
    diff = pos - B(t_closest)
    tangent = B_prime(t_closest)
    normal = np.array([-tangent[1], tangent[0]])
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    return np.sign(np.dot(diff, normal)) * np.linalg.norm(diff), t_closest

def compute_heading_error(theta, B_prime, t_closest):
    tangent = B_prime(t_closest)
    error = theta - np.arctan2(tangent[1], tangent[0])
    return np.arctan2(np.sin(error), np.cos(error))


# ==============================================================================
# PART 4: Simulation
# ==============================================================================

def simulate_bezier_following(config=None, verbose=True):
    print("\n" + "=" * 80)
    print("  Multi-Articulated Vehicle MPC Simulation")
    print("  Targets: T1 Rear, SU1, SU2, T2 Rear (4-point tracking)")
    print("  ★ HIGH-PERFORMANCE VERSION (nlpsol + CodeGen + batch Bezier)")
    print("=" * 80)
    
    mpc = MultiArticulatedVehicleMPC()
    metrics = PerformanceMetrics()
    
    if config:
        bezier_points, obstacles_list, safety_margin = config['control_points'], config['obstacles'], config['safety_margin']
    else:
        bezier_points = [[0, 0], [15, 0], [15, 30], [30, 30]]
        obstacles_list = [{'center': [17, 22], 'radius': 2.0}]
        safety_margin = 1.2
    
    mpc.vehicle_safety_radius = safety_margin
    B, B_prime, B_batch = get_bezier_functions(bezier_points)
    s_to_t, t_to_s, total_length, s_to_t_batch, s_samples_tbl, t_samples_tbl = create_arc_length_interpolator(B_prime)
    obs_params = [obstacles_list[0]['center'][0], obstacles_list[0]['center'][1], obstacles_list[0]['radius']] if obstacles_list else [1000, 1000, 1.0]
    
    if obstacles_list:
        print(f"  Path: {len(bezier_points)} pts, {total_length:.2f}m | Obstacle: ({obs_params[0]:.1f},{obs_params[1]:.1f}), r={obs_params[2]:.1f}m")
    else:
        print(f"  Path: {len(bezier_points)} pts, {total_length:.2f}m | No obstacles")
    
    s_init, lat_err_init = 7.0, -0.5
    x0 = initialize_vehicle_along_bezier(B, B_prime, s_to_t, s_init, mpc, lat_err_init, 0.0)
    
    print(f"  Initial: s={s_init:.1f}m, lat_err={lat_err_init:.2f}m")
    
    v_max, accel_time, decel_accel = 3.0, 5.0, 0.2
    T_sim, dt, steps = 60.0, mpc.dt, int(60.0 / mpc.dt)
    
    X_history, U_history = np.zeros((11, steps + 1)), np.zeros((5, steps))
    X_history[:, 0] = x0
    x_current, X_guess, U_guess = x0.copy(), None, None
    u_prev, u_prev_prev = np.zeros(mpc.nu), np.zeros(mpc.nu)
    
    leader_trajectory = []  # [(x, y, theta, s), ...]
    leader_s_values = []    # ★ バイナリサーチ用の弧長リスト
    leader_x_values = []
    leader_y_values = []
    
    # ========================================
    # ★ 幾何学的な遅れ距離の計算 (Arc Length Delay)
    # ========================================
    # T1_Rear -> Hitch1 -> Steer1
    dist_T1_to_SU1 = mpc.R1 + mpc.L2 + mpc.L3
    
    # SU1 -> Conn1 -> Conn2 -> SU2
    # Note: SU1 to Conn1 is L3 backward? No, Conn1 = Steer1 + L3.
    # Geometry:
    # T1R --(R1)--> Hitch1 --(L2+L3)--> Steer1
    # Steer1 --(L3)--> Conn1 --(L4)--> Conn2 --(L5)--> Steer2
    # Steer2 --(L6+L5)--> Hitch2 --(R2)--> T2R
    
    # Path Following logic assumes we follow the trace of T1 Rear.
    # So we need the distance along the "string" of the vehicle.
    # SU1 is approx (R1 + L2 + L3) behind T1R.
    # SU2 is approx (L3 + L4 + L5) behind SU1.
    # T2R is approx (L6 + L5 + R2) behind SU2.
    
    dist_SU1_to_SU2 = mpc.L3 + mpc.L4 + mpc.L5
    dist_SU2_to_T2 = mpc.L6 + mpc.L5 + mpc.R2
    
    # total_dist_T1_to_SU2 = dist_T1_to_SU1 + dist_SU1_to_SU2
    # total_dist_T1_to_T2 = total_dist_T1_to_SU2 + dist_SU2_to_T2
    total_dist_T1_to_SU2 = mpc.R1 + mpc.L2 + mpc.L4  - mpc.L5
    total_dist_T1_to_T2 = total_dist_T1_to_SU2 + mpc.L5 + mpc.L6 + mpc.R2
    
    # 重み設定 (4点に対応)
    weights = {
        'Q_pos_t1': 10.0,      # T1 Rear
        'Q_pos_su1': 5.0,      # SU1 (was Cargo)
        'Q_pos_su2': 5.0,      # SU2 (New)
        'Q_pos_t2': 10.0,      # T2 Rear
        'Q_angle': 1.0,
        'Q_vel': 1.0,
        'R': 1.0,
        'R_rate': 10.0,
        'Q_phi_rate': 20.0,
        'Q_phi_mag': 1.0
    }
    
    def get_position_from_trajectory(target_s):
        """★ バイナリサーチ版 (O(log n))"""
        if not leader_s_values:
            return B(s_to_t(max(0, target_s)))
        idx = bisect.bisect_left(leader_s_values, target_s)
        if idx == 0:
            best_idx = 0
        elif idx >= len(leader_s_values):
            best_idx = len(leader_s_values) - 1
        else:
            if abs(leader_s_values[idx] - target_s) < abs(leader_s_values[idx-1] - target_s):
                best_idx = idx
            else:
                best_idx = idx - 1
        if abs(leader_s_values[best_idx] - target_s) < 0.5:
            return np.array([leader_x_values[best_idx], leader_y_values[best_idx]])
        else:
            return B(s_to_t(max(0, target_s)))
    
    if verbose:
        print("\n Step |  s[m] | v_tgt | Lat_T1 | Lat_SU1| Lat_SU2| Lat_T2 | MinClr | Solve | Avg_ms")
        print("-" * 95)
    
    t_hint = None  # ★ 最近接点探索のヒント
    
    for i in range(steps):
        t1_rear = x_current[0:2]
        theta_t1 = x_current[2]
        t_on_curve = find_closest_point(t1_rear, B, t_hint=t_hint)
        t_hint = t_on_curve  # ★ 次回のヒント更新
        s_current = t_to_s(t_on_curve)
        v_target = calculate_target_velocity(i * dt, s_current, total_length, v_max, accel_time, decel_accel)
        
        leader_trajectory.append((t1_rear[0], t1_rear[1], theta_t1, s_current))
        leader_s_values.append(s_current)
        leader_x_values.append(t1_rear[0])
        leader_y_values.append(t1_rear[1])
        
        # ★ 参照軌道生成: T1はバッチ処理で高速化
        t1_ref = np.zeros((2, mpc.N + 1))
        su1_ref = np.zeros((2, mpc.N + 1))
        su2_ref = np.zeros((2, mpc.N + 1))
        t2_ref = np.zeros((2, mpc.N + 1))
        v_ref_array = np.full(mpc.N, v_target)
        
        # T1参照のバッチ計算
        k_range = np.arange(mpc.N + 1)
        s_future = np.minimum(s_current + k_range * dt * v_target, total_length)
        t_future = s_to_t_batch(s_future)
        if B_batch is not None:
            t1_ref = B_batch(t_future).T  # (2, N+1)
        else:
            for k in range(mpc.N + 1):
                t1_ref[:, k] = B(t_future[k])
        
        for k in range(mpc.N + 1):
            s_fut = s_future[k]
            
            # SU1: T1から遅れ
            s_su1 = s_fut - dist_T1_to_SU1
            su1_ref[:, k] = get_position_from_trajectory(s_su1) if s_su1 > 0 and leader_trajectory else B(s_to_t(max(0, s_su1)))
            
            # SU2: T1からさらに遅れ
            s_su2 = s_fut - total_dist_T1_to_SU2
            su2_ref[:, k] = get_position_from_trajectory(s_su2) if s_su2 > 0 and leader_trajectory else B(s_to_t(max(0, s_su2)))
            
            # T2 Rear: T1からさらに遅れ
            s_t2 = s_fut - total_dist_T1_to_T2
            t2_ref[:, k] = get_position_from_trajectory(s_t2) if s_t2 > 0 and leader_trajectory else B(s_to_t(max(0, s_t2)))
        
        # ウォームスタート初期化
        if X_guess is None or U_guess is None:
            X_guess = np.tile(x_current[:, None], (1, mpc.N + 1))
            U_guess = np.zeros((mpc.nu, mpc.N))
        
        # MPC Solve
        u_opt, X_plan, U_plan, t_solve, success = mpc.solve(
            x_current, t1_ref, su1_ref, su2_ref, t2_ref, obs_params, v_ref_array, weights, X_guess, U_guess
        )
        
        # 現在位置の計算 (4点)
        p_t1, p_su1, p_su2, p_t2 = compute_vehicle_positions_numpy(x_current, mpc.L1, mpc.L2, mpc.L3, mpc.L4, mpc.L5, mpc.L6, mpc.L7, mpc.R1, mpc.R2)
        
        # 誤差計算
        lat_t1, t_cl_t1 = compute_lateral_error(p_t1, B, B_prime)
        lat_su1, _ = compute_lateral_error(p_su1, B, B_prime)
        lat_su2, _ = compute_lateral_error(p_su2, B, B_prime)
        lat_t2, _ = compute_lateral_error(p_t2, B, B_prime)
        
        metrics.lat_error_t1.append(lat_t1)
        metrics.lat_error_su1.append(lat_su1)
        metrics.lat_error_su2.append(lat_su2)
        metrics.lat_error_t2.append(lat_t2)
        
        metrics.heading_error_t1.append(compute_heading_error(x_current[2], B_prime, t_cl_t1))
        
        obs_c = np.array(obs_params[0:2])
        # 4点それぞれの障害物距離
        dist_list = [np.linalg.norm(p - obs_c) for p in [p_t1, p_su1, p_su2, p_t2]]
        clr = min(dist_list) - obs_params[2] - safety_margin
        
        metrics.min_obstacle_dist.append(clr)
        metrics.obstacle_clearance_t1.append(dist_list[0] - obs_params[2] - safety_margin)
        metrics.obstacle_clearance_su1.append(dist_list[1] - obs_params[2] - safety_margin)
        metrics.obstacle_clearance_su2.append(dist_list[2] - obs_params[2] - safety_margin)
        metrics.obstacle_clearance_t2.append(dist_list[3] - obs_params[2] - safety_margin)
        
        if i >= 2:
            metrics.velocity_jerk.append((u_opt[0] - 2*u_prev[0] + u_prev_prev[0]) / dt**2)
            metrics.steering_jerk.append((u_opt[1] - 2*u_prev[1] + u_prev_prev[1]) / dt**2)
        
        metrics.solve_times.append(t_solve)
        metrics.solver_success.append(success)
        metrics.arc_length_progress.append(s_current)
        metrics.velocity_actual.append(u_opt[0] if success else 0)
        metrics.velocity_target.append(v_target)
        metrics.control_mode.append("NORM")
        
        if success:
            X_guess = np.roll(X_plan, -1, axis=1)
            U_guess = np.roll(U_plan, -1, axis=1)
            X_guess[:, -1] = X_plan[:, -1]
            U_guess[:, -1] = U_plan[:, -1]
        else:
            if X_guess is not None:
                X_guess = np.roll(X_guess, -1, axis=1)
                U_guess = np.roll(U_guess, -1, axis=1)
            u_opt = U_history[:, i-1] if i > 0 else np.array([v_target, 0, 0, 0, 0])
        
        U_history[:, i] = u_opt
        x_current = np.array(mpc.F(x_current, u_opt)).flatten()
        u_prev_prev, u_prev = u_prev.copy(), u_opt.copy()
        X_history[:, i + 1] = x_current
        
        if verbose and i % 20 == 0:
            avg_solve = np.mean(metrics.solve_times) if metrics.solve_times else 0.0
            print(f" {i:4d} | {s_current:5.1f} | {v_target:5.2f} | {lat_t1:+6.3f} | {lat_su1:+6.3f} | {lat_su2:+6.3f} | {lat_t2:+6.3f} | {clr:+6.2f} | {t_solve:5.1f} | {avg_solve:6.1f}")
        
        if s_current >= total_length * 0.98 and v_target < 0.05:
            print(f"\n  GOAL at step {i+1}")
            X_history, U_history = X_history[:, :i+2], U_history[:, :i+1]
            break
    
    metrics.print_summary(dt)
    return X_history, U_history, mpc, metrics, bezier_points, obs_params

# ==============================================================================
# PART 5: Plotting & Export (機能完全版)
# ==============================================================================

def compute_slip_angles(X_history, U_history, mpc, dt):
    """
    4点 (T1, SU1, SU2, T2) のスリップ角を計算
    """
    slip_angles = {'t1': [], 'su1': [], 'su2': [], 't2': []}
    L1, L2, L3, L4, L5, L6, L7, R1, R2 = mpc.L1, mpc.L2, mpc.L3, mpc.L4, mpc.L5, mpc.L6, mpc.L7, mpc.R1, mpc.R2
    
    for i in range(X_history.shape[1] - 1):
        x_curr = X_history[:, i]
        x_next = X_history[:, i + 1]
        
        # 現在位置と次ステップ位置の計算
        p1_c, su1_c, su2_c, p2_c = compute_vehicle_positions_numpy(x_curr, L1, L2, L3, L4, L5, L6, L7, R1, R2)
        p1_n, su1_n, su2_n, p2_n = compute_vehicle_positions_numpy(x_next, L1, L2, L3, L4, L5, L6, L7, R1, R2)
        
        # 各点の角度（モデル定義に基づく）
        # T1: theta_a1(idx2), SU1: theta_a2(idx4), SU2: theta_a6(idx8), T2: theta_a7(idx9)
        thetas = [x_curr[2], x_curr[4], x_curr[8], x_curr[9]]
        pos_currs = [p1_c, su1_c, su2_c, p2_c]
        pos_nexts = [p1_n, su1_n, su2_n, p2_n]
        keys = ['t1', 'su1', 'su2', 't2']
        
        for key, theta, p_c, p_n in zip(keys, thetas, pos_currs, pos_nexts):
            dx = p_n[0] - p_c[0]
            dy = p_n[1] - p_c[1]
            speed = np.sqrt(dx**2 + dy**2)
            
            if speed > 1e-6:
                # 速度ベクトルの角度
                vel_angle = np.arctan2(dy, dx)
                # スリップ角 = 速度方向 - 車体の向き
                slip = np.arctan2(np.sin(vel_angle - theta), np.cos(vel_angle - theta))
            else:
                slip = 0.0
            slip_angles[key].append(slip)
            
    return {k: np.array(v) for k, v in slip_angles.items()}

def save_figure(fig, output_dir, filename_base, save_eps=True):
    if save_eps:
        eps_path = os.path.join(output_dir, f'{filename_base}.eps')
        png_path = os.path.join(output_dir, f'{filename_base}.png')
        fig.savefig(eps_path, format='eps', bbox_inches='tight', dpi=300)
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        print(f"    Saved: {filename_base}.eps, {filename_base}.png")

# ============================================================================
# 4点追従用 plot_simulation_graphs 改善版（完全版）
# 3点追従レベルの詳細な出力機能を追加
# ============================================================================

def plot_simulation_graphs(X_history, U_history, metrics, dt, mpc, bezier_points, obs_params, save_eps=True, output_dir='./'):
    """
    シミュレーション結果の包括的な可視化
    
    Parameters:
    -----------
    X_history : np.ndarray, shape (11, N)
        状態履歴
    U_history : np.ndarray, shape (5, N-1)
        制御入力履歴
    metrics : PerformanceMetrics
        パフォーマンスメトリクス
    dt : float
        時間刻み
    mpc : MPCController
        MPCコントローラインスタンス
    bezier_points : np.ndarray
        ベジェ曲線の制御点
    obs_params : tuple
        障害物パラメータ (x, y, radius)
    save_eps : bool
        EPS形式で保存するか
    output_dir : str
        出力ディレクトリ
    
    Returns:
    --------
    slip_data : dict
        スリップ角データ
    """
    print("\n  Generating comprehensive plots...")
    
    def save_figure(fig, output_dir, name, save_eps):
        """図を保存"""
        if save_eps:
            eps_path = os.path.join(output_dir, f'{name}.eps')
            fig.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
            print(f"    Saved: {eps_path}")
        png_path = os.path.join(output_dir, f'{name}.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {png_path}")
    
    # 時間軸の作成
    time_ax = np.arange(len(metrics.lat_error_t1)) * dt
    time_u = np.arange(U_history.shape[1]) * dt
    time_x = np.arange(X_history.shape[1]) * dt
    
    # ベジェ曲線の参照パス生成
    bezier_points_array = np.array(bezier_points)  # リストをnumpy配列に変換
    t_vals = np.linspace(0, 1, 500)
    n = len(bezier_points_array) - 1
    ref_path = np.zeros((len(t_vals), 2))
    for i, t in enumerate(t_vals):
        for j in range(n + 1):
            b = comb(n, j) * ((1 - t)**(n - j)) * (t**j)
            ref_path[i] += bezier_points_array[j] * b
    
    # 車体位置履歴の計算
    t1_hist, su1_hist, su2_hist, t2_hist = [], [], [], []
    for i in range(X_history.shape[1]):
        p1, p_su1, p_su2, p2 = compute_vehicle_positions_numpy(
            X_history[:, i], mpc.L1, mpc.L2, mpc.L3, mpc.L4, 
            mpc.L5, mpc.L6, mpc.L7, mpc.R1, mpc.R2
        )
        t1_hist.append(p1)
        su1_hist.append(p_su1)
        su2_hist.append(p_su2)
        t2_hist.append(p2)
    
    t1_hist = np.array(t1_hist)
    su1_hist = np.array(su1_hist)
    su2_hist = np.array(su2_hist)
    t2_hist = np.array(t2_hist)
    
    # スリップ角の計算
    slip_data = compute_slip_angles(X_history, U_history, mpc, dt)
    time_slip = np.arange(len(slip_data['t1'])) * dt
    
    # ========================================
    # Figure 1: Trajectory (軌跡)
    # ========================================
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.plot(ref_path[:, 0], ref_path[:, 1], 'k--', lw=2, label='Reference path')
    ax1.plot(t1_hist[:, 0], t1_hist[:, 1], 'b-', lw=1.5, label='Tractor 1 Rear')
    ax1.plot(su1_hist[:, 0], su1_hist[:, 1], 'g-', lw=1.5, label='Steering Unit 1')
    ax1.plot(su2_hist[:, 0], su2_hist[:, 1], 'c-', lw=1.5, label='Steering Unit 2')
    ax1.plot(t2_hist[:, 0], t2_hist[:, 1], 'r-', lw=1.5, label='Tractor 2 Rear')
    
    # 障害物（障害物が実際に存在する場合のみ描画）
    if obs_params[0] < 100:  # 障害物が実際に存在する場合（x座標が100未満）
        obs_circle = Circle((obs_params[0], obs_params[1]), obs_params[2], 
                            color='orange', alpha=0.5, label='Obstacle')
        ax1.add_patch(obs_circle)
        safety_circle = Circle((obs_params[0], obs_params[1]), 
                              obs_params[2] + mpc.vehicle_safety_radius, 
                              color='darkorange', fill=False, ls='--', lw=1.5, 
                              label='Safety margin')
        ax1.add_patch(safety_circle)
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(fig1, output_dir, 'fig_trajectory', save_eps)
    plt.show()
    
    # ========================================
    # Figure 2: Lateral Error (横偏差) - 4点を個別にプロット
    # ========================================
    # T1 (Tractor 1)
    fig2a, ax2a = plt.subplots(figsize=(6, 4))
    ax2a.plot(time_ax, np.array(metrics.lat_error_t1) * 100, 'r-', lw=1.5)
    ax2a.axhline(0, color='k', ls='-', lw=0.5)
    ax2a.set_xlabel('Time [s]')
    ax2a.set_ylabel('Lateral error [cm]')
    ax2a.grid(True, alpha=0.3)
    ax2a.set_xlim([0, time_ax[-1]])
    plt.tight_layout()
    save_figure(fig2a, output_dir, 'fig_lateral_error_t1', save_eps)
    plt.close(fig2a)
    
    # SU1 (Steering Unit 1)
    fig2b, ax2b = plt.subplots(figsize=(6, 4))
    ax2b.plot(time_ax, np.array(metrics.lat_error_su1) * 100, 'r-', lw=1.5)
    ax2b.axhline(0, color='k', ls='-', lw=0.5)
    ax2b.set_xlabel('Time [s]')
    ax2b.set_ylabel('Lateral error [cm]')
    ax2b.grid(True, alpha=0.3)
    ax2b.set_xlim([0, time_ax[-1]])
    plt.tight_layout()
    save_figure(fig2b, output_dir, 'fig_lateral_error_su1', save_eps)
    plt.close(fig2b)
    
    # SU2 (Steering Unit 2)
    fig2c, ax2c = plt.subplots(figsize=(6, 4))
    ax2c.plot(time_ax, np.array(metrics.lat_error_su2) * 100, 'r-', lw=1.5)
    ax2c.axhline(0, color='k', ls='-', lw=0.5)
    ax2c.set_xlabel('Time [s]')
    ax2c.set_ylabel('Lateral error [cm]')
    ax2c.grid(True, alpha=0.3)
    ax2c.set_xlim([0, time_ax[-1]])
    plt.tight_layout()
    save_figure(fig2c, output_dir, 'fig_lateral_error_su2', save_eps)
    plt.close(fig2c)
    
    # T2 (Tractor 2)
    fig2d, ax2d = plt.subplots(figsize=(6, 4))
    ax2d.plot(time_ax, np.array(metrics.lat_error_t2) * 100, 'r-', lw=1.5)
    ax2d.axhline(0, color='k', ls='-', lw=0.5)
    ax2d.set_xlabel('Time [s]')
    ax2d.set_ylabel('Lateral error [cm]')
    ax2d.grid(True, alpha=0.3)
    ax2d.set_xlim([0, time_ax[-1]])
    plt.tight_layout()
    save_figure(fig2d, output_dir, 'fig_lateral_error_t2', save_eps)
    plt.close(fig2d)

    # ========================================
    # Figure 3: Heading Error (方位偏差)
    # ========================================
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(time_ax, np.rad2deg(metrics.heading_error_t1), 'r-', lw=1.5)
    ax3.axhline(0, color='k', ls='-', lw=0.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Heading error [deg]')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, time_ax[-1]])
    plt.tight_layout()
    
    save_figure(fig3, output_dir, 'fig_heading_error', save_eps)
    plt.close(fig3)
    
    # ========================================
    # Figure 4: Obstacle Clearance (障害物クリアランス) - 障害物がある場合のみ
    # ========================================
    if obs_params[0] < 100:  # 障害物が実際に存在する場合のみ作成
        # T1 clearance
        fig4a, ax4a = plt.subplots(figsize=(6, 4))
        ax4a.plot(time_ax, metrics.obstacle_clearance_t1, 'r-', lw=1.5)
        ax4a.axhline(0, color='k', ls='--', lw=1.5, label='Safety limit')
        ax4a.set_xlabel('Time [s]')
        ax4a.set_ylabel('Clearance [m]')
        ax4a.legend(loc='best', framealpha=0.9)
        ax4a.grid(True, alpha=0.3)
        ax4a.set_xlim([0, time_ax[-1]])
        plt.tight_layout()
        save_figure(fig4a, output_dir, 'fig_obstacle_clearance_t1', save_eps)
        plt.close(fig4a)
        
        # SU1 clearance
        fig4b, ax4b = plt.subplots(figsize=(6, 4))
        ax4b.plot(time_ax, metrics.obstacle_clearance_su1, 'r-', lw=1.5)
        ax4b.axhline(0, color='k', ls='--', lw=1.5, label='Safety limit')
        ax4b.set_xlabel('Time [s]')
        ax4b.set_ylabel('Clearance [m]')
        ax4b.legend(loc='best', framealpha=0.9)
        ax4b.grid(True, alpha=0.3)
        ax4b.set_xlim([0, time_ax[-1]])
        plt.tight_layout()
        save_figure(fig4b, output_dir, 'fig_obstacle_clearance_su1', save_eps)
        plt.close(fig4b)
        
        # SU2 clearance
        fig4c, ax4c = plt.subplots(figsize=(6, 4))
        ax4c.plot(time_ax, metrics.obstacle_clearance_su2, 'r-', lw=1.5)
        ax4c.axhline(0, color='k', ls='--', lw=1.5, label='Safety limit')
        ax4c.set_xlabel('Time [s]')
        ax4c.set_ylabel('Clearance [m]')
        ax4c.legend(loc='best', framealpha=0.9)
        ax4c.grid(True, alpha=0.3)
        ax4c.set_xlim([0, time_ax[-1]])
        plt.tight_layout()
        save_figure(fig4c, output_dir, 'fig_obstacle_clearance_su2', save_eps)
        plt.close(fig4c)
        
        # T2 clearance
        fig4d, ax4d = plt.subplots(figsize=(6, 4))
        ax4d.plot(time_ax, metrics.obstacle_clearance_t2, 'r-', lw=1.5)
        ax4d.axhline(0, color='k', ls='--', lw=1.5, label='Safety limit')
        ax4d.set_xlabel('Time [s]')
        ax4d.set_ylabel('Clearance [m]')
        ax4d.legend(loc='best', framealpha=0.9)
        ax4d.grid(True, alpha=0.3)
        ax4d.set_xlim([0, time_ax[-1]])
        plt.tight_layout()
        save_figure(fig4d, output_dir, 'fig_obstacle_clearance_t2', save_eps)
        plt.close(fig4d)
    else:
        print("  [Skip] Obstacle clearance plot (No obstacles)")


    # ========================================
    # Figure 5: Slip Angle (スリップ角)
    # ========================================
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    ax5.plot(time_slip, np.rad2deg(slip_data['t1']), 'b-', lw=1.5, label='Tractor 1')
    ax5.plot(time_slip, np.rad2deg(slip_data['su1']), 'g-', lw=1.5, label='Steering Unit 1')
    ax5.plot(time_slip, np.rad2deg(slip_data['su2']), 'c-', lw=1.5, label='Steering Unit 2')
    ax5.plot(time_slip, np.rad2deg(slip_data['t2']), 'r-', lw=1.5, label='Tractor 2')
    ax5.axhline(0, color='k', ls='-', lw=0.5)
    ax5.fill_between(time_slip, -1, 1, alpha=0.2, color='green', label='Good zone (±1 deg)')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Slip Angle [deg]')
    ax5.legend(loc='upper right', framealpha=0.9)
    ax5.set_title('Slip Angle Verification')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, time_slip[-1]])
    plt.tight_layout()
    
    save_figure(fig5, output_dir, 'fig_slip_angle', save_eps)
    plt.show()
    
    # スリップ角の統計を出力
    print("\n  --- Slip Angle Analysis (スリップ角検証) ---")
    for key, name in [('t1', 'Tractor 1'), ('su1', 'Steering Unit 1'), 
                      ('su2', 'Steering Unit 2'), ('t2', 'Tractor 2')]:
        deg_data = np.rad2deg(slip_data[key])
        print(f"    {name:18s}: Max={np.max(np.abs(deg_data)):7.4f}deg, "
              f"Mean={np.mean(np.abs(deg_data)):7.4f}deg, "
              f"Std={np.std(deg_data):7.4f}deg")

    # ========================================
    # Figure 6: Velocity (速度)
    # ========================================
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    ax6.plot(time_u, U_history[0, :], 'r-', lw=1.5, label='Actual')
    if metrics.velocity_target:
        time_target = time_ax[:len(metrics.velocity_target)]
        ax6.plot(time_target, metrics.velocity_target, 'b-', lw=1.5, label='Target')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Velocity [m/s]')
    ax6.legend(loc='best', framealpha=0.9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([0, time_u[-1]])
    plt.tight_layout()
    
    save_figure(fig6, output_dir, 'fig_velocity', save_eps)
    plt.close(fig6)
    
    # ========================================
    # Figure 7: Steering Rate (舵角速度) - 前後に分離
    # ========================================
    # Front steering rate
    fig7a, ax7a = plt.subplots(figsize=(6, 4))
    ax7a.plot(time_u, np.rad2deg(U_history[1, :]), 'r-', lw=1.5)
    ax7a.axhline(0, color='k', ls='-', lw=0.5)
    ax7a.set_xlabel('Time [s]')
    ax7a.set_ylabel('Steering rate [deg/s]')
    ax7a.grid(True, alpha=0.3)
    ax7a.set_xlim([0, time_u[-1]])
    plt.tight_layout()
    save_figure(fig7a, output_dir, 'fig_steering_rate_front', save_eps)
    plt.close(fig7a)
    
    # Rear steering rate
    fig7b, ax7b = plt.subplots(figsize=(6, 4))
    ax7b.plot(time_u, np.rad2deg(U_history[4, :]), 'r-', lw=1.5)
    ax7b.axhline(0, color='k', ls='-', lw=0.5)
    ax7b.set_xlabel('Time [s]')
    ax7b.set_ylabel('Steering rate [deg/s]')
    ax7b.grid(True, alpha=0.3)
    ax7b.set_xlim([0, time_u[-1]])
    plt.tight_layout()
    save_figure(fig7b, output_dir, 'fig_steering_rate_rear', save_eps)
    plt.close(fig7b)
    
    # ========================================
    # Figure 8: Computation Time (計算時間)
    # ========================================
    fig8, ax8 = plt.subplots(figsize=(8, 4))
    ax8.plot(np.arange(len(metrics.solve_times)), metrics.solve_times, 'b-', lw=1)
    ax8.axhline(dt * 1000, color='r', ls='--', lw=2, 
                label=f'Real-time limit ({dt*1000:.0f} ms)')
    ax8.axhline(np.mean(metrics.solve_times), color='g', ls='-', lw=2, 
                label=f'Mean ({np.mean(metrics.solve_times):.1f} ms)')
    ax8.set_xlabel('Step')
    ax8.set_ylabel('Solve time [ms]')
    ax8.legend(loc='upper right', framealpha=0.9)
    ax8.set_title('MPC Computation Time')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([0, len(metrics.solve_times)])
    plt.tight_layout()
    
    save_figure(fig8, output_dir, 'fig_computation_time', save_eps)
    plt.show()
    
    # ========================================
    # Figure 9: Steering Angles (舵角)
    # ========================================
    fig9, ax9 = plt.subplots(figsize=(8, 4))
    ax9.plot(time_x, np.rad2deg(X_history[3, :]), 'b-', lw=1.5, label='Front steering $\phi_1$')
    ax9.plot(time_x, np.rad2deg(X_history[10, :]), 'r-', lw=1.5, label='Rear steering $\phi_2$')
    ax9.axhline(0, color='k', ls='-', lw=0.5)
    ax9.axhline(np.rad2deg(mpc.phi_max), color='k', ls='--', lw=1, alpha=0.5)
    ax9.axhline(-np.rad2deg(mpc.phi_max), color='k', ls='--', lw=1, alpha=0.5)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Steering angle [deg]')
    ax9.legend(loc='upper right', framealpha=0.9)
    ax9.set_title('Steering Angles')
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim([0, time_x[-1]])
    plt.tight_layout()
    
    save_figure(fig9, output_dir, 'fig_steering_angles', save_eps)
    plt.show()
    
    # ========================================
    # Figure 10: Lateral Slip Displacement (横滑り変位)
    # ========================================
    lateral_slip_t1_list = []
    lateral_slip_su1_list = []
    lateral_slip_su2_list = []
    lateral_slip_t2_list = []
    
    prev_pos_t1 = None
    prev_pos_su1 = None
    prev_pos_su2 = None
    prev_pos_t2 = None
    
    for i in range(X_history.shape[1]):
        x = X_history[:, i]
        L1, L2, L3, L4 = mpc.L1, mpc.L2, mpc.L3, mpc.L4
        L5, L6, L7 = mpc.L5, mpc.L6, mpc.L7
        R1, R2 = mpc.R1, mpc.R2
        
        # 各車体の位置を計算
        p_t1, p_su1, p_su2, p_t2 = compute_vehicle_positions_numpy(
            x, L1, L2, L3, L4, L5, L6, L7, R1, R2
        )
        
        if i > 0 and prev_pos_t1 is not None:
            # Tractor 1 Rear
            dx_t1 = p_t1[0] - prev_pos_t1[0]
            dy_t1 = p_t1[1] - prev_pos_t1[1]
            lateral_slip_t1 = -dx_t1 * np.sin(x[2]) + dy_t1 * np.cos(x[2])
            
            # Steering Unit 1
            dx_su1 = p_su1[0] - prev_pos_su1[0]
            dy_su1 = p_su1[1] - prev_pos_su1[1]
            lateral_slip_su1 = -dx_su1 * np.sin(x[4]) + dy_su1 * np.cos(x[4])
            
            # Steering Unit 2
            dx_su2 = p_su2[0] - prev_pos_su2[0]
            dy_su2 = p_su2[1] - prev_pos_su2[1]
            lateral_slip_su2 = -dx_su2 * np.sin(x[8]) + dy_su2 * np.cos(x[8])
            
            # Tractor 2 Rear
            dx_t2 = p_t2[0] - prev_pos_t2[0]
            dy_t2 = p_t2[1] - prev_pos_t2[1]
            lateral_slip_t2 = -dx_t2 * np.sin(x[9]) + dy_t2 * np.cos(x[9])
        else:
            lateral_slip_t1 = 0.0
            lateral_slip_su1 = 0.0
            lateral_slip_su2 = 0.0
            lateral_slip_t2 = 0.0
        
        lateral_slip_t1_list.append(lateral_slip_t1)
        lateral_slip_su1_list.append(lateral_slip_su1)
        lateral_slip_su2_list.append(lateral_slip_su2)
        lateral_slip_t2_list.append(lateral_slip_t2)
        
        prev_pos_t1 = p_t1.copy()
        prev_pos_su1 = p_su1.copy()
        prev_pos_su2 = p_su2.copy()
        prev_pos_t2 = p_t2.copy()
    
    time_lateral = np.arange(len(lateral_slip_t1_list)) * dt
    
    # T1 lateral slip
    fig10a, ax10a = plt.subplots(figsize=(6, 4))
    ax10a.plot(time_lateral, np.array(lateral_slip_t1_list) * 1000, 'r-', lw=1.5)
    ax10a.axhline(0, color='k', ls='-', lw=0.5)
    ax10a.set_xlabel('Time [s]')
    ax10a.set_ylabel('Lateral slip [mm]')
    ax10a.grid(True, alpha=0.3)
    ax10a.set_xlim([0, time_lateral[-1]])
    plt.tight_layout()
    save_figure(fig10a, output_dir, 'fig_lateral_slip_t1', save_eps)
    plt.close(fig10a)
    
    # SU1 lateral slip
    fig10b, ax10b = plt.subplots(figsize=(6, 4))
    ax10b.plot(time_lateral, np.array(lateral_slip_su1_list) * 1000, 'r-', lw=1.5)
    ax10b.axhline(0, color='k', ls='-', lw=0.5)
    ax10b.set_xlabel('Time [s]')
    ax10b.set_ylabel('Lateral slip [mm]')
    ax10b.grid(True, alpha=0.3)
    ax10b.set_xlim([0, time_lateral[-1]])
    plt.tight_layout()
    save_figure(fig10b, output_dir, 'fig_lateral_slip_su1', save_eps)
    plt.close(fig10b)
    
    # SU2 lateral slip
    fig10c, ax10c = plt.subplots(figsize=(6, 4))
    ax10c.plot(time_lateral, np.array(lateral_slip_su2_list) * 1000, 'r-', lw=1.5)
    ax10c.axhline(0, color='k', ls='-', lw=0.5)
    ax10c.set_xlabel('Time [s]')
    ax10c.set_ylabel('Lateral slip [mm]')
    ax10c.grid(True, alpha=0.3)
    ax10c.set_xlim([0, time_lateral[-1]])
    plt.tight_layout()
    save_figure(fig10c, output_dir, 'fig_lateral_slip_su2', save_eps)
    plt.close(fig10c)
    
    # T2 lateral slip
    fig10d, ax10d = plt.subplots(figsize=(6, 4))
    ax10d.plot(time_lateral, np.array(lateral_slip_t2_list) * 1000, 'r-', lw=1.5)
    ax10d.axhline(0, color='k', ls='-', lw=0.5)
    ax10d.set_xlabel('Time [s]')
    ax10d.set_ylabel('Lateral slip [mm]')
    ax10d.grid(True, alpha=0.3)
    ax10d.set_xlim([0, time_lateral[-1]])
    plt.tight_layout()
    save_figure(fig10d, output_dir, 'fig_lateral_slip_t2', save_eps)
    plt.close(fig10d)
    
    # 横滑り変位の統計を出力
    print("\n  --- Lateral Slip Displacement Analysis (横滑り変位検証) ---")
    print(f"    Tractor 1:      Max={np.max(np.abs(lateral_slip_t1_list))*1000:8.4f}mm, "
          f"Mean={np.mean(np.abs(lateral_slip_t1_list))*1000:8.4f}mm, "
          f"RMSE={np.sqrt(np.mean(np.array(lateral_slip_t1_list)**2))*1000:8.4f}mm")
    print(f"    Steer Unit 1:   Max={np.max(np.abs(lateral_slip_su1_list))*1000:8.4f}mm, "
          f"Mean={np.mean(np.abs(lateral_slip_su1_list))*1000:8.4f}mm, "
          f"RMSE={np.sqrt(np.mean(np.array(lateral_slip_su1_list)**2))*1000:8.4f}mm")
    print(f"    Steer Unit 2:   Max={np.max(np.abs(lateral_slip_su2_list))*1000:8.4f}mm, "
          f"Mean={np.mean(np.abs(lateral_slip_su2_list))*1000:8.4f}mm, "
          f"RMSE={np.sqrt(np.mean(np.array(lateral_slip_su2_list)**2))*1000:8.4f}mm")
    print(f"    Tractor 2:      Max={np.max(np.abs(lateral_slip_t2_list))*1000:8.4f}mm, "
          f"Mean={np.mean(np.abs(lateral_slip_t2_list))*1000:8.4f}mm, "
          f"RMSE={np.sqrt(np.mean(np.array(lateral_slip_t2_list)**2))*1000:8.4f}mm")
    
    # ========================================
    # Figure 11+: State Variables (状態変数)
    # ========================================
    state_names = ["x", "y", "theta1", "phi1", "theta2", "theta3", 
                   "theta4", "theta5", "theta6", "theta7", "phi2"]
    state_labels = {
        "x": "x [m]",
        "y": "y [m]",
        "theta1": r"$\theta_1$ [rad]",
        "phi1": r"$\phi_1$ [rad]",
        "theta2": r"$\theta_2$ [rad]",
        "theta3": r"$\theta_3$ [rad]",
        "theta4": r"$\theta_4$ [rad]",
        "theta5": r"$\theta_5$ [rad]",
        "theta6": r"$\theta_6$ [rad]",
        "theta7": r"$\theta_7$ [rad]",
        "phi2": r"$\phi_2$ [rad]"
    }
    
    for i, name in enumerate(state_names):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_x, X_history[i, :], 'r-', lw=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(state_labels[name])
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_x[-1]])
        # x-y座標の場合はアスペクト比を1:1に
        if name in ["x", "y"]:
            ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        save_figure(fig, output_dir, f'fig_state_{name}_with_obs', save_eps)
        plt.close(fig)
    
    print("\n  All figures generated successfully!")
    
    return slip_data

def calculate_wheel_rotations_and_export(X_history, U_history, mpc, obs_params, bezier_pts, metrics, filename='mpc_simulation_results.csv'):
    print("\n  Exporting CSV for C++ Viewer (Full Bezier Path)...")
    
    # 1. 時間補間
    dt_orig = mpc.dt
    time_orig = np.arange(X_history.shape[1]) * dt_orig
    
    fps = 60.0 
    dt_new = 1.0 / fps
    time_new = np.arange(0, time_orig[-1], dt_new)
    steps_new = len(time_new)
    
    # 状態変数の補間
    X_interp = np.zeros((X_history.shape[0], steps_new))
    angle_indices = [2, 4, 5, 6, 7, 8, 9] 
    for i in range(X_history.shape[0]):
        val = X_history[i, :]
        if i in angle_indices:
            val = np.unwrap(val)
        X_interp[i, :] = interp1d(time_orig, val, fill_value="extrapolate")(time_new)
    
    # 入力の補間
    U_padded = np.hstack([U_history, U_history[:, -1:]]) 
    U_interp = np.zeros((U_history.shape[0], steps_new))
    for i in range(U_history.shape[0]):
        U_interp[i, :] = interp1d(time_orig, U_padded[i, :], fill_value="extrapolate")(time_new)
    
    # ベジェ曲線関数の取得
    B, B_prime, B_batch_csv = get_bezier_functions(bezier_pts)
    
    # 【修正箇所1】 CSVの行数に合わせて、0.0〜1.0 (始点〜終点) を等分割したtを作成
    t_for_csv = np.linspace(0, 1.0, steps_new)
    
    L1, L2, L3, L4, L5, L6, L7 = mpc.L1, mpc.L2, mpc.L3, mpc.L4, mpc.L5, mpc.L6, mpc.L7
    R1, R2 = mpc.R1, mpc.R2
    wheel_radius = 0.2
    
    data_list = []
    spin_t1_f = 0.0; spin_t1_r = 0.0; spin_t2_f = 0.0; spin_t2_r = 0.0
    prev_pos = None; prev_points_for_slip = None
    
    for i in range(steps_new):
        x = X_interp[:, i]
        u = U_interp[:, i]
        
        # 位置計算 (4点)
        p_t1, p_su1, p_su2, p_t2 = compute_vehicle_positions_numpy(x, L1, L2, L3, L4, L5, L6, L7, R1, R2)
        
        t1_front_pos = p_t1 + L1 * np.array([np.cos(x[2]), np.sin(x[2])])
        t2_front_pos = p_t2 + L7 * np.array([np.cos(x[9]), np.sin(x[9])])
        
        curr_pos_wheel = {'t1_r': p_t1, 't1_f': t1_front_pos, 't2_r': p_t2, 't2_f': t2_front_pos}
        curr_points_for_slip = [p_t1, p_su1, p_su2, p_t2]
        
        if prev_pos is not None:
            direction = 1.0 if u[0] >= 0 else -1.0
            spin_t1_f += direction * np.linalg.norm(curr_pos_wheel['t1_f'] - prev_pos['t1_f']) / wheel_radius
            spin_t1_r += direction * np.linalg.norm(curr_pos_wheel['t1_r'] - prev_pos['t1_r']) / wheel_radius
            spin_t2_f += direction * np.linalg.norm(curr_pos_wheel['t2_f'] - prev_pos['t2_f']) / wheel_radius
            spin_t2_r += direction * np.linalg.norm(curr_pos_wheel['t2_r'] - prev_pos['t2_r']) / wheel_radius
        prev_pos = curr_pos_wheel
        
        lat_slips = [0.0] * 4
        if i > 0 and prev_points_for_slip is not None:
            angles = [x[2], x[4], x[8], x[9]] 
            for idx in range(4):
                dx = curr_points_for_slip[idx][0] - prev_points_for_slip[idx][0]
                dy = curr_points_for_slip[idx][1] - prev_points_for_slip[idx][1]
                lat_slips[idx] = -dx * np.sin(angles[idx]) + dy * np.cos(angles[idx])
        prev_points_for_slip = curr_points_for_slip
        
        # 【修正箇所2】 車両位置ではなく、t_for_csv[i] を使ってベジェ座標を計算
        bz_pos = B(t_for_csv[i])
        bz_tan = B_prime(t_for_csv[i])
        bz_theta = np.arctan2(bz_tan[1], bz_tan[0])
        
        # 解析用データ（こちらは車両位置ベースで計算）
        eps_t1, t_cl_t1 = compute_lateral_error(p_t1, B, B_prime)
        eps_su1, _ = compute_lateral_error(p_su1, B, B_prime)
        eps_su2, _ = compute_lateral_error(p_su2, B, B_prime)
        eps_t2, _ = compute_lateral_error(p_t2, B, B_prime)
        heading_err_t1 = compute_heading_error(x[2], B_prime, t_cl_t1)
        
        row = {}
        row['time'] = time_new[i]
        row['x'] = x[0]; row['y'] = x[1]; row['theta1'] = x[2]; row['phi1'] = x[3]
        row['theta2'] = x[4]; row['theta3'] = x[5]; row['theta4'] = x[6]; row['theta5'] = x[7]
        row['theta6'] = x[8]; row['theta7'] = x[9]; row['phi2'] = x[10]
        row['v1'] = u[0]; row['v2'] = u[1]; row['v3'] = u[2]; row['v4'] = u[3]; row['v5'] = u[4]
        row['tireFL'] = spin_t1_f; row['tireFR'] = spin_t1_f; row['tireRL'] = spin_t1_r; row['tireRR'] = spin_t1_r
        row['trailFL'] = spin_t2_f; row['trailFR'] = spin_t2_f; row['trailRL'] = spin_t2_r; row['trailRR'] = spin_t2_r
        
        # 修正されたベジェ座標
        row['bezier_x'] = bz_pos[0]; row['bezier_y'] = bz_pos[1]; row['bezier_theta'] = bz_theta
        
        row['obs_x'] = obs_params[0]; row['obs_y'] = obs_params[1]; row['obs_r'] = obs_params[2]
        
        row['eps_t1'] = eps_t1; row['eps_su1'] = eps_su1; row['eps_su2'] = eps_su2; row['eps_t2'] = eps_t2
        row['heading_err_t1'] = heading_err_t1
        row['lat_slip_t1'] = lat_slips[0]; row['lat_slip_su1'] = lat_slips[1]
        row['lat_slip_su2'] = lat_slips[2]; row['lat_slip_t2'] = lat_slips[3]
        
        data_list.append(row)
        
    cols = [
        'time', 'x', 'y', 'theta1', 'phi1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'phi2',
        'v1', 'v2', 'v3', 'v4', 'v5',
        'tireFL', 'tireFR', 'tireRL', 'tireRR', 'trailFL', 'trailFR', 'trailRL', 'trailRR',
        'bezier_x', 'bezier_y', 'bezier_theta',
        'obs_x', 'obs_y', 'obs_r',
        'eps_t1', 'eps_su1', 'eps_su2', 'eps_t2', 'heading_err_t1',
        'lat_slip_t1', 'lat_slip_su1', 'lat_slip_su2', 'lat_slip_t2'
    ]
    
    df = pd.DataFrame(data_list, columns=cols)
    df.to_csv(filename, index=False)
    print(f"  Saved CSV compatible with C++ viewer: {os.path.abspath(filename)}")
    
    lateral_slip_data = {
        't1': df['lat_slip_t1'].values,
        'su1': df['lat_slip_su1'].values,
        'su2': df['lat_slip_su2'].values,
        't2': df['lat_slip_t2'].values
    }
    return lateral_slip_data

# ============================================================================
# 4点追従用 export_metrics_json 関数改善版（完全版）
# 3点追従レベルの詳細なJSON出力とコンソール表示を追加
# ============================================================================

import json
import numpy as np

def export_metrics_json(metrics, slip_data, lateral_slip_data, filename='mpc_metrics.json'):

        # ===== コンソールにサマリー出力 =====
        print("\n" + "=" * 80)
        print("  SIMULATION RESULTS SUMMARY")
        print("=" * 80)
        
        # 追従精度サマリー
        metrics.print_summary(0.1)
        
        # 横滑り変位サマリー
        print("\n--- LATERAL SLIP DISPLACEMENT ---")
        for key, values in lateral_slip_data.items():
            values_m = np.array(values) * 1000  # m → mm変換
            max_slip = np.max(np.abs(values_m))
            mean_slip = np.mean(np.abs(values_m))
            rmse_slip = np.sqrt(np.mean(values_m**2))
            print(f"  {key.upper():4s}: Max={max_slip:.4f}mm, Mean={mean_slip:.4f}mm, RMSE={rmse_slip:.4f}mm")
        
        # リアルタイム係数の計算
        if metrics.solve_times:
            mean_solve_time_ms = np.mean(metrics.solve_times)
            dt_ms = 0.1 * 1000
            rt_factor = dt_ms / mean_solve_time_ms
            rt_status = "CAPABLE" if rt_factor > 1.0 else "NOT CAPABLE"
            print(f"\n--- REAL-TIME PERFORMANCE ---")
            print(f"  Sampling Time: {dt_ms:.1f}ms")
            print(f"  Mean Solve Time: {mean_solve_time_ms:.2f}ms")
            print(f"  Real-time Factor: {rt_factor:.2f}x ({rt_status})")
        
        print("=" * 80)
        

def export_metrics_json(metrics, slip_data, lateral_slip_data, filename='mpc_metrics.json'):
    """
    メトリクスをJSON形式でエクスポート（詳細版）
    
    Parameters:
    -----------
    metrics : PerformanceMetrics
        パフォーマンスメトリクス
    slip_data : dict
        スリップ角データ {'t1': [...], 'su1': [...], 'su2': [...], 't2': [...]}
    lateral_slip_data : dict
        横滑り変位データ {'t1': [...], 'su1': [...], 'su2': [...], 't2': [...]}
    filename : str
        出力ファイル名
    """
    stats = metrics.compute_statistics()
    
    # スリップ角の統計を追加
    if slip_data:
        stats['slip_angle'] = {}
        for key, name in [('t1', 'tractor1'), ('su1', 'steering_unit1'), 
                          ('su2', 'steering_unit2'), ('t2', 'tractor2')]:
            deg_data = np.rad2deg(slip_data[key])
            stats['slip_angle'][name] = {
                'max_deg': float(np.max(np.abs(deg_data))),
                'mean_deg': float(np.mean(np.abs(deg_data))),
                'std_deg': float(np.std(deg_data))
            }
    
    # 横滑り変位の統計を追加
    if lateral_slip_data:
        stats['lateral_slip_displacement'] = {}
        for key, name in [('t1', 'tractor1'), ('su1', 'steering_unit1'), 
                          ('su2', 'steering_unit2'), ('t2', 'tractor2')]:
            data = lateral_slip_data[key]
            stats['lateral_slip_displacement'][name] = {
                'max_m': float(np.max(np.abs(data))),
                'mean_m': float(np.mean(np.abs(data))),
                'std_m': float(np.std(data)),
                'rmse_m': float(np.sqrt(np.mean(data**2)))
            }
        
        # 横滑り変位の統計をコンソールに表示
        print("\n--- LATERAL SLIP DISPLACEMENT ---")
        print(f"  Tractor 1:      RMSE={stats['lateral_slip_displacement']['tractor1']['rmse_m']:.6f}m, "
              f"Max={stats['lateral_slip_displacement']['tractor1']['max_m']:.6f}m, "
              f"Mean(abs)={stats['lateral_slip_displacement']['tractor1']['mean_m']:.6f}m")
        print(f"  Steering Unit 1: RMSE={stats['lateral_slip_displacement']['steering_unit1']['rmse_m']:.6f}m, "
              f"Max={stats['lateral_slip_displacement']['steering_unit1']['max_m']:.6f}m, "
              f"Mean(abs)={stats['lateral_slip_displacement']['steering_unit1']['mean_m']:.6f}m")
        print(f"  Steering Unit 2: RMSE={stats['lateral_slip_displacement']['steering_unit2']['rmse_m']:.6f}m, "
              f"Max={stats['lateral_slip_displacement']['steering_unit2']['max_m']:.6f}m, "
              f"Mean(abs)={stats['lateral_slip_displacement']['steering_unit2']['mean_m']:.6f}m")
        print(f"  Tractor 2:      RMSE={stats['lateral_slip_displacement']['tractor2']['rmse_m']:.6f}m, "
              f"Max={stats['lateral_slip_displacement']['tractor2']['max_m']:.6f}m, "
              f"Mean(abs)={stats['lateral_slip_displacement']['tractor2']['mean_m']:.6f}m")
    
    # JSON出力データの構築
    data = {
        'statistics': stats,
        'time_series': {
            'lat_error_t1': metrics.lat_error_t1,
            'lat_error_su1': metrics.lat_error_su1,
            'lat_error_su2': metrics.lat_error_su2,
            'lat_error_t2': metrics.lat_error_t2,
            'heading_error_t1': [float(x) for x in metrics.heading_error_t1],
            'obstacle_clearance': metrics.min_obstacle_dist,
            'solve_times': metrics.solve_times,
            'velocity_actual': metrics.velocity_actual,
            'velocity_target': metrics.velocity_target,
            'control_mode': metrics.control_mode,
            'slip_angle_t1': [float(x) for x in slip_data['t1']] if slip_data else [],
            'slip_angle_su1': [float(x) for x in slip_data['su1']] if slip_data else [],
            'slip_angle_su2': [float(x) for x in slip_data['su2']] if slip_data else [],
            'slip_angle_t2': [float(x) for x in slip_data['t2']] if slip_data else [],
            'lateral_slip_t1': [float(x) for x in lateral_slip_data['t1']] if lateral_slip_data else [],
            'lateral_slip_su1': [float(x) for x in lateral_slip_data['su1']] if lateral_slip_data else [],
            'lateral_slip_su2': [float(x) for x in lateral_slip_data['su2']] if lateral_slip_data else [],
            'lateral_slip_t2': [float(x) for x in lateral_slip_data['t2']] if lateral_slip_data else []
        }
    }
    
    # JSON出力
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"    Saved: {filename}")

# ============================================================================
# メイン処理の統一版（3点追従・4点追従共通）
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("  Bezier MPC Simulation - WITH OBSTACLES VERSION (HIGH-PERFORMANCE)")
    print("  Optimizations: nlpsol + CodeGen + batch Bezier + bisect trajectory + warm start")
    print("=" * 80 + "\n")
    
    # 初期設定（障害物あり）
    initial_points = [[0, 0], [30, 0], [20, 10], [10, 20], [0, 30], [30, 30]]
    initial_obstacles = [{'center': [16, 15], 'radius': 2.0}]
    config_result = {'config': None}
    
    # 直接設定を使用（障害物あり）
    config_result['config'] = {
        'control_points': initial_points,
        'obstacles': initial_obstacles,  # 障害物あり
        'safety_margin': 1.2
    }
    
    print("  Mode: Direct simulation WITH obstacles (Simple graph style)")
    print(f"  Control points: {len(initial_points)} points")
    print(f"  Obstacles: {len(initial_obstacles)} - Center: {initial_obstacles[0]['center']}, Radius: {initial_obstacles[0]['radius']}m\n")
    
    if config_result['config']:
        # 実行時刻でフォルダを作成
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, f"results_with_obstacles_dt01_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n  Output directory: {output_dir}")
        
        # MPCシミュレーション実行
        X_hist, U_hist, mpc_inst, metrics, bezier_pts, obs_params = simulate_bezier_following(
            config_result['config'], verbose=True
        )
        
        # グラフ生成（スリップ角データ取得）
        slip_data = plot_simulation_graphs(
            X_hist, U_hist, metrics, mpc_inst.dt, mpc_inst, bezier_pts, obs_params,
            save_eps=True, output_dir=output_dir
        )
        
        # CSVエクスポート（横滑り変位データ取得）
        lateral_slip_data = calculate_wheel_rotations_and_export(
            X_hist, U_hist, mpc_inst, obs_params, bezier_pts, metrics,
            os.path.join(output_dir, 'mpc_simulation_results.csv')
        )
        
        # JSONエクスポート（統計データと時系列データ）
        export_metrics_json(
            metrics, slip_data, lateral_slip_data,
            os.path.join(output_dir, 'mpc_metrics.json')
        )
        
        # 設定情報を保存
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_result['config'], f, indent=2)
        print(f"    Saved: {config_path}")
        
        # 完了メッセージ
        print("\n" + "=" * 80)
        print("  ✓ Simulation Complete!")
        print(f"  Output directory: {output_dir}")
        print("  Files:")
        print("    - mpc_simulation_results.csv  (C++ Viewer + Analysis data)")
        print("    - mpc_metrics.json            (Statistics + Time series)")
        print("    - config.json                 (Simulation configuration)")
        print("    - fig_*.eps                   (Vector graphics for papers)")
        print("    - fig_*.png                   (Raster graphics for preview)")
        print("=" * 80 + "\n")
    else:
        print("\n  Cancelled.")
