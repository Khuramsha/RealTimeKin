import numpy as np


def parse_sensor_names(header_text):
    fields = [f for f in header_text.strip().split('\t') if f]
    if not fields:
        return []
    if fields[0].lower() == 'time':
        fields = fields[1:]
    return fields


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / np.dot(q, q)


def quat_log(q):
    q = quat_normalize(q)
    w, x, y, z = q
    v = np.array([x, y, z])
    v_norm = np.linalg.norm(v)
    w = np.clip(w, -1.0, 1.0)
    if v_norm < 1e-12:
        return 2.0 * v
    angle = 2.0 * np.arctan2(v_norm, w)
    axis = v / v_norm
    return axis * angle


def quat_from_rot_matrix(R):
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_yaw(angle):
    half = 0.5 * angle
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0])


class HeadingCorrector:
    def __init__(
        self,
        sensor_names,
        model,
        rate,
        update_hz=10.0,
        pelvis_gain=0.02,
        leg_gain=0.08,
        yaw_limit=np.deg2rad(20.0),
        yaw_outlier=np.deg2rad(35.0),
        stance_gyro_thresh=np.deg2rad(40.0),
        use_bias=False,
    ):
        self.sensor_names = sensor_names
        self.sensor_indices = {name: idx for idx, name in enumerate(sensor_names)}
        self.rate = rate
        self.update_period = 1.0 / update_hz
        self.last_update_time = -np.inf
        self.use_bias = use_bias
        self.yaw_offsets = {'P': 0.0, 'L': 0.0, 'R': 0.0}
        self.yaw_rate_bias = {'P': 0.0, 'L': 0.0, 'R': 0.0}
        self.pelvis_gain = pelvis_gain
        self.leg_gain = leg_gain
        self.yaw_limit = yaw_limit
        self.yaw_outlier = yaw_outlier
        self.stance_gyro_thresh = stance_gyro_thresh
        self.z_hat = np.array([0.0, 1.0, 0.0])
        self.prev_quat = {}
        self.stance_state = {'L': False, 'R': False}
        self.sensor_groups = {
            'P': ['pelvis_imu'],
            'L': ['femur_l_imu', 'tibia_l_imu', 'calcn_l_imu'],
            'R': ['femur_r_imu', 'tibia_r_imu', 'calcn_r_imu'],
        }
        self.sensor_body_map = {
            'pelvis_imu': ('pelvis', 'pelvis_imu'),
            'femur_l_imu': ('femur_l', None),
            'tibia_l_imu': ('tibia_l', None),
            'calcn_l_imu': ('calcn_l', None),
            'femur_r_imu': ('femur_r', None),
            'tibia_r_imu': ('tibia_r', None),
            'calcn_r_imu': ('calcn_r', None),
        }
        self.sensor_frame_quat = self._cache_sensor_frames(model)

    def _cache_sensor_frames(self, model):
        frame_quat = {}
        for name, (body_name, frame_name) in self.sensor_body_map.items():
            q_bs = np.array([1.0, 0.0, 0.0, 0.0])
            if frame_name is not None:
                frame_path = f"/bodyset/{body_name}/{frame_name}"
                try:
                    frame = model.getComponent(frame_path)
                    rot = frame.getTransformInParent().R()
                    R = np.array([
                        [rot.get(0, 0), rot.get(0, 1), rot.get(0, 2)],
                        [rot.get(1, 0), rot.get(1, 1), rot.get(1, 2)],
                        [rot.get(2, 0), rot.get(2, 1), rot.get(2, 2)],
                    ])
                    q_bs = quat_from_rot_matrix(R)
                except Exception:
                    q_bs = np.array([1.0, 0.0, 0.0, 0.0])
            frame_quat[name] = q_bs
        return frame_quat

    def apply(self, q_meas):
        q_corr = np.zeros_like(q_meas)
        for name, idx in self.sensor_indices.items():
            group = self._group_for_sensor(name)
            delta = self.yaw_offsets.get(group, 0.0)
            q_corr[idx] = quat_mul(quat_yaw(delta), q_meas[idx])
        return q_corr

    def update_stance(self, q_meas, dt):
        for side, foot_name in [('L', 'calcn_l_imu'), ('R', 'calcn_r_imu')]:
            if foot_name not in self.sensor_indices:
                continue
            idx = self.sensor_indices[foot_name]
            q_curr = quat_normalize(q_meas[idx])
            q_prev = self.prev_quat.get(foot_name, q_curr)
            q_delta = quat_mul(quat_inv(q_prev), q_curr)
            rot_vec = quat_log(q_delta)
            ang_vel = np.linalg.norm(rot_vec) / max(dt, 1e-6)
            self.stance_state[side] = ang_vel < self.stance_gyro_thresh
            self.prev_quat[foot_name] = q_curr

    def compute_model_quats(self, model, state):
        model_quats = {}
        for name, (body_name, _) in self.sensor_body_map.items():
            if name not in self.sensor_indices:
                continue
            try:
                body = model.getBodySet().get(body_name)
            except Exception:
                continue
            rot = body.getTransformInGround(state).R()
            R = np.array([
                [rot.get(0, 0), rot.get(0, 1), rot.get(0, 2)],
                [rot.get(1, 0), rot.get(1, 1), rot.get(1, 2)],
                [rot.get(2, 0), rot.get(2, 1), rot.get(2, 2)],
            ])
            q_gb = quat_from_rot_matrix(R)
            q_bs = self.sensor_frame_quat.get(name, np.array([1.0, 0.0, 0.0, 0.0]))
            model_quats[name] = quat_mul(q_gb, q_bs)
        return model_quats

    def update(self, q_meas, q_corr, q_model, time_s, dt, ik_success=True, residual_metric=None):
        if not ik_success:
            return
        if residual_metric is not None and not np.isfinite(residual_metric):
            return
        if residual_metric is not None and residual_metric > self.yaw_outlier:
            return
        if (time_s - self.last_update_time) < self.update_period:
            return

        self.update_stance(q_meas, dt)

        for group, sensors in self.sensor_groups.items():
            if group == 'L' and not self.stance_state.get('L', False):
                continue
            if group == 'R' and not self.stance_state.get('R', False):
                continue
            e_yaws = []
            for sensor in sensors:
                if sensor not in self.sensor_indices:
                    continue
                idx = self.sensor_indices[sensor]
                if sensor not in q_model:
                    continue
                q_err = quat_mul(q_model[sensor], quat_inv(quat_normalize(q_corr[idx])))
                e_vec = quat_log(q_err)
                e_yaw = float(np.dot(e_vec, self.z_hat))
                e_yaws.append(e_yaw)
            if not e_yaws:
                continue
            e_med = float(np.median(e_yaws))
            if abs(e_med) > self.yaw_outlier:
                continue
            gain = self.pelvis_gain if group == 'P' else self.leg_gain
            if self.use_bias:
                self.yaw_rate_bias[group] += -gain * e_med / max(dt, 1e-6)
                self.yaw_offsets[group] += self.yaw_rate_bias[group] * dt
            else:
                self.yaw_offsets[group] += -gain * e_med
            self.yaw_offsets[group] = float(
                np.clip(self.yaw_offsets[group], -self.yaw_limit, self.yaw_limit)
            )

        self.last_update_time = time_s

    def _group_for_sensor(self, name):
        for group, sensors in self.sensor_groups.items():
            if name in sensors:
                return group
        return 'P'
