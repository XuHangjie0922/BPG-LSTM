import torch
from torch import nn
import torch.nn.functional as F


class FixedPINNLoss(nn.Module):
    def __init__(
        self,
        mono_weight: float = 0.1,
        smooth_weight: float = 0.01,
        failure_weight: float = 0.1,
        failure_threshold: float = 5.0,
    ) -> None:
        super().__init__()
        self.mono_weight = mono_weight
        self.smooth_weight = smooth_weight
        self.failure_weight = failure_weight
        self.failure_threshold = failure_threshold
        self.supervised_loss = nn.MSELoss()

    def forward(self, pred_seq: torch.Tensor, target_seq: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pred_last = pred_seq[:, -1]
        target_last = target_seq[:, -1]

        rul_loss = self.supervised_loss(pred_last, target_last)
        mono_loss = torch.relu(pred_seq[:, 1:] - pred_seq[:, :-1]).mean()
        smooth_loss = ((pred_seq[:, 1:] - pred_seq[:, :-1]) + 1.0).pow(2).mean()

        near_failure = target_last <= self.failure_threshold
        if near_failure.any():
            failure_loss = pred_last[near_failure].pow(2).mean()
        else:
            failure_loss = pred_last.new_tensor(0.0)

        total = (
            rul_loss
            + self.mono_weight * mono_loss
            + self.smooth_weight * smooth_loss
            + self.failure_weight * failure_loss
        )
        parts = {
            "total_loss": float(total.detach().cpu()),
            "rul_loss": float(rul_loss.detach().cpu()),
            "mono_loss": float(mono_loss.detach().cpu()),
            "smooth_loss": float(smooth_loss.detach().cpu()),
            "failure_loss": float(failure_loss.detach().cpu()),
        }
        return total, parts


class BRDRWeightBalancer:
    def __init__(
        self,
        names: list[str],
        beta_c: float = 0.99,
        beta_w: float = 0.95,
        eps: float = 1e-12,
        min_weight: float = 0.25,
        max_weight: float = 4.0,
        weight_power: float = 0.75,
        warmup_steps: int = 20,
    ) -> None:
        self.names = names
        self.beta_c = beta_c
        self.beta_w = beta_w
        self.eps = eps
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_power = weight_power
        self.warmup_steps = warmup_steps
        self.loss_square_ema = {name: 0.0 for name in names}
        self.updates = {name: 0 for name in names}
        self.weights = {name: 1.0 for name in names}
        self.steps = 0

    def update(
        self,
        values: dict[str, torch.Tensor],
        active_names: set[str] | None = None,
    ) -> dict[str, float]:
        self.steps += 1
        if self.steps <= self.warmup_steps:
            return dict(self.weights)

        if active_names is None:
            active_names = set(values)
        else:
            active_names = set(active_names)
        active_names = {name for name in active_names if name in self.names and name in values}
        if not active_names:
            return dict(self.weights)

        for name in self.names:
            if name not in active_names:
                continue
            loss_value = max(float(values[name].detach().cpu()), 0.0)
            self.updates[name] += 1
            self.loss_square_ema[name] = (
                self.beta_c * self.loss_square_ema[name] + (1.0 - self.beta_c) * (loss_value**2)
            )

        decay_rates = {}
        for name in active_names:
            bias_correction = max(1.0 - self.beta_c ** self.updates[name], self.eps)
            loss_square_mean = self.loss_square_ema[name] / bias_correction
            loss_value = max(float(values[name].detach().cpu()), 0.0)
            decay_rates[name] = loss_value / ((loss_square_mean + self.eps) ** 0.5)

        mean_rate = sum(decay_rates.values()) / len(decay_rates)
        for name in active_names:
            target_weight = decay_rates[name] / max(mean_rate, self.eps)
            target_weight = target_weight ** self.weight_power
            target_weight = min(max(target_weight, self.min_weight), self.max_weight)
            self.weights[name] = self.beta_w * self.weights[name] + (1.0 - self.beta_w) * target_weight

        mean_weight = sum(self.weights.values()) / len(self.weights)
        for name in self.names:
            self.weights[name] /= max(mean_weight, self.eps)
            self.weights[name] = min(max(self.weights[name], self.min_weight), self.max_weight)

        mean_weight = sum(self.weights.values()) / len(self.weights)
        for name in self.names:
            self.weights[name] /= max(mean_weight, self.eps)

        return dict(self.weights)

    def state_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "beta_c": self.beta_c,
            "beta_w": self.beta_w,
            "eps": self.eps,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "weight_power": self.weight_power,
            "warmup_steps": self.warmup_steps,
            "loss_square_ema": dict(self.loss_square_ema),
            "updates": dict(self.updates),
            "weights": dict(self.weights),
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.loss_square_ema.update({str(k): float(v) for k, v in dict(state.get("loss_square_ema", {})).items()})
        self.updates.update({str(k): int(v) for k, v in dict(state.get("updates", {})).items()})
        self.weights.update({str(k): float(v) for k, v in dict(state.get("weights", {})).items()})
        self.steps = int(state.get("steps", self.steps))


class BRDRPINNLoss(nn.Module):
    def __init__(
        self,
        rul_weight: float = 1.0,
        mono_weight: float = 0.1,
        smooth_weight: float = 0.01,
        failure_weight: float = 0.1,
        failure_threshold: float = 5.0,
        beta_c: float = 0.99,
        beta_w: float = 0.95,
        adaptive_rul: bool = False,
        brdr_min_weight: float = 0.25,
        brdr_max_weight: float = 4.0,
        brdr_weight_power: float = 0.75,
        brdr_warmup_steps: int = 20,
    ) -> None:
        super().__init__()
        self.base_weights = {
            "rul_loss": rul_weight,
            "mono_loss": mono_weight,
            "smooth_loss": smooth_weight,
            "failure_loss": failure_weight,
        }
        self.failure_threshold = failure_threshold
        self.adaptive_rul = adaptive_rul
        self.supervised_loss = nn.MSELoss()
        adaptive_names = ["mono_loss", "smooth_loss", "failure_loss"]
        if adaptive_rul:
            adaptive_names = ["rul_loss"] + adaptive_names
        self.balancer = BRDRWeightBalancer(
            names=adaptive_names,
            beta_c=beta_c,
            beta_w=beta_w,
            min_weight=brdr_min_weight,
            max_weight=brdr_max_weight,
            weight_power=brdr_weight_power,
            warmup_steps=brdr_warmup_steps,
        )

    def forward(self, pred_seq: torch.Tensor, target_seq: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        pred_last = pred_seq[:, -1]
        target_last = target_seq[:, -1]

        component_losses = {
            "rul_loss": self.supervised_loss(pred_last, target_last),
            "mono_loss": torch.relu(pred_seq[:, 1:] - pred_seq[:, :-1]).mean(),
            "smooth_loss": ((pred_seq[:, 1:] - pred_seq[:, :-1]) + 1.0).pow(2).mean(),
        }

        near_failure = target_last <= self.failure_threshold
        if near_failure.any():
            component_losses["failure_loss"] = pred_last[near_failure].pow(2).mean()
        else:
            component_losses["failure_loss"] = pred_last.new_tensor(0.0)

        active_names = set(self.balancer.names)
        if not near_failure.any():
            active_names.discard("failure_loss")
        adaptive_weights = self.balancer.update(
            {name: component_losses[name] for name in self.balancer.names},
            active_names=active_names,
        )
        total = pred_seq.new_tensor(0.0)
        for name, loss in component_losses.items():
            adaptive_weight = adaptive_weights.get(name, 1.0)
            total = total + self.base_weights[name] * adaptive_weight * loss

        parts = {
            "total_loss": float(total.detach().cpu()),
        }
        for name, loss in component_losses.items():
            parts[name] = float(loss.detach().cpu())
            parts[f"w_{name}"] = float(adaptive_weights.get(name, 1.0))
        return total, parts


class ConditionAwareBRDRPINNLoss(nn.Module):
    def __init__(
        self,
        num_conditions: int,
        rul_weight: float = 1.0,
        mono_weight: float = 0.1,
        smooth_weight: float = 0.01,
        failure_weight: float = 0.1,
        failure_threshold: float = 5.0,
        beta_c: float = 0.99,
        beta_w: float = 0.95,
        brdr_min_weight: float = 0.25,
        brdr_max_weight: float = 4.0,
        brdr_weight_power: float = 0.75,
        brdr_warmup_steps: int = 20,
    ) -> None:
        super().__init__()
        self.num_conditions = num_conditions
        self.base_weights = {
            "rul_loss": rul_weight,
            "mono_loss": mono_weight,
            "smooth_loss": smooth_weight,
            "failure_loss": failure_weight,
        }
        self.failure_threshold = failure_threshold
        self.physics_balancer = BRDRWeightBalancer(
            names=["mono_loss", "smooth_loss", "failure_loss"],
            beta_c=beta_c,
            beta_w=beta_w,
            min_weight=brdr_min_weight,
            max_weight=brdr_max_weight,
            weight_power=brdr_weight_power,
            warmup_steps=brdr_warmup_steps,
        )
        self.condition_balancer = BRDRWeightBalancer(
            names=[f"condition_{i}" for i in range(num_conditions)],
            beta_c=beta_c,
            beta_w=beta_w,
            min_weight=brdr_min_weight,
            max_weight=brdr_max_weight,
            weight_power=brdr_weight_power,
            warmup_steps=brdr_warmup_steps,
        )

    def forward(
        self,
        pred_seq: torch.Tensor,
        target_seq: torch.Tensor,
        condition_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred_last = pred_seq[:, -1]
        target_last = target_seq[:, -1]
        condition_ids = condition_ids.long()

        sample_rul_sq = (pred_last - target_last).pow(2)
        batch_rul_loss = sample_rul_sq.mean()

        condition_values = {}
        for condition_idx in range(self.num_conditions):
            mask = condition_ids == condition_idx
            if mask.any():
                condition_values[f"condition_{condition_idx}"] = sample_rul_sq[mask].mean()
        condition_weights = self.condition_balancer.update(
            condition_values,
            active_names=set(condition_values),
        )

        sample_weights = pred_last.new_ones(len(pred_last))
        for condition_idx in range(self.num_conditions):
            mask = condition_ids == condition_idx
            if mask.any():
                sample_weights[mask] = condition_weights[f"condition_{condition_idx}"]
        rul_loss = (sample_weights * sample_rul_sq).mean()

        component_losses = {
            "mono_loss": torch.relu(pred_seq[:, 1:] - pred_seq[:, :-1]).mean(),
            "smooth_loss": ((pred_seq[:, 1:] - pred_seq[:, :-1]) + 1.0).pow(2).mean(),
        }

        near_failure = target_last <= self.failure_threshold
        if near_failure.any():
            component_losses["failure_loss"] = pred_last[near_failure].pow(2).mean()
        else:
            component_losses["failure_loss"] = pred_last.new_tensor(0.0)

        active_physics_names = set(component_losses)
        if not near_failure.any():
            active_physics_names.discard("failure_loss")
        physics_weights = self.physics_balancer.update(
            component_losses,
            active_names=active_physics_names,
        )

        total = self.base_weights["rul_loss"] * rul_loss
        for name, loss in component_losses.items():
            total = total + self.base_weights[name] * physics_weights.get(name, 1.0) * loss

        parts = {
            "total_loss": float(total.detach().cpu()),
            "rul_loss": float(rul_loss.detach().cpu()),
            "raw_rul_loss": float(batch_rul_loss.detach().cpu()),
            "mono_loss": float(component_losses["mono_loss"].detach().cpu()),
            "smooth_loss": float(component_losses["smooth_loss"].detach().cpu()),
            "failure_loss": float(component_losses["failure_loss"].detach().cpu()),
            "w_rul_loss": 1.0,
            "w_mono_loss": float(physics_weights["mono_loss"]),
            "w_smooth_loss": float(physics_weights["smooth_loss"]),
            "w_failure_loss": float(physics_weights["failure_loss"]),
        }
        for condition_idx in range(self.num_conditions):
            parts[f"w_condition_{condition_idx}"] = float(condition_weights[f"condition_{condition_idx}"])
        return total, parts


class StageConditionAwareBRDRPINNLoss(nn.Module):
    def __init__(
        self,
        num_conditions: int,
        num_stages: int = 3,
        max_rul: float = 125.0,
        rul_weight: float = 1.0,
        mono_weight: float = 0.1,
        smooth_weight: float = 0.01,
        rul_dynamics_weight: float = 0.0,
        rul_dynamics_threshold: float = 80.0,
        sensor_trend_weight: float = 0.0,
        sensor_trend_signs: torch.Tensor | None = None,
        sensor_trend_min_stage: int = 1,
        sensor_trend_huber_delta: float = 1.0,
        health_weight: float = 0.0,
        health_mono_weight: float = 0.0,
        weak_health_index: bool = False,
        health_margin: float = 0.01,
        hi_min_std: float = 0.05,
        hi_endpoint_gap: float = 0.2,
        hi_range_eps: float = 0.02,
        health_low: torch.Tensor | None = None,
        health_high: torch.Tensor | None = None,
        condition_sensor_baseline: torch.Tensor | None = None,
        middle_rul: float = 100.0,
        late_rul: float = 40.0,
        failure_weight: float = 0.1,
        asym_weight: float = 0.0,
        asym_over_alpha: float = 10.0,
        asym_under_beta: float = 13.0,
        asym_clip: float = 50.0,
        over_margin_weight: float = 0.0,
        over_margin: float = 0.0,
        late_stage_weight: float = 1.5,
        learnable_stage_prior: bool = False,
        stage_prior_reg_weight: float = 0.001,
        stage_prior_range: float = 0.75,
        failure_threshold: float = 5.0,
        beta_c: float = 0.99,
        beta_w: float = 0.95,
        brdr_min_weight: float = 0.25,
        brdr_max_weight: float = 4.0,
        brdr_weight_power: float = 0.75,
        brdr_warmup_steps: int = 20,
    ) -> None:
        super().__init__()
        self.num_conditions = num_conditions
        self.num_stages = num_stages
        self.max_rul = max_rul
        self.rul_dynamics_threshold = rul_dynamics_threshold
        self.middle_rul = middle_rul
        self.late_rul = late_rul
        if sensor_trend_signs is None:
            sensor_trend_signs = torch.zeros(1, dtype=torch.float32)
        self.register_buffer("sensor_trend_signs", sensor_trend_signs.float())
        self.sensor_trend_min_stage = max(int(sensor_trend_min_stage), 0)
        self.sensor_trend_huber_delta = max(float(sensor_trend_huber_delta), 1e-6)
        self.weak_health_index = bool(weak_health_index)
        self.health_margin = max(float(health_margin), 0.0)
        self.hi_min_std = max(float(hi_min_std), 0.0)
        self.hi_endpoint_gap = max(float(hi_endpoint_gap), 0.0)
        self.hi_range_eps = min(max(float(hi_range_eps), 0.0), 0.49)
        if health_low is None:
            health_low = torch.zeros(num_conditions, dtype=torch.float32)
        if health_high is None:
            health_high = torch.ones(num_conditions, dtype=torch.float32)
        self.register_buffer("health_low", health_low.float())
        self.register_buffer("health_high", health_high.float())
        if condition_sensor_baseline is None:
            condition_sensor_baseline = torch.zeros(num_conditions, sensor_trend_signs.shape[-1])
        self.register_buffer("condition_sensor_baseline", condition_sensor_baseline.float())
        self.base_weights = {
            "rul_loss": rul_weight,
            "mono_loss": mono_weight,
            "smooth_loss": smooth_weight,
            "rul_dynamics_loss": rul_dynamics_weight,
            "sensor_trend_loss": sensor_trend_weight,
            "health_loss": health_weight,
            "health_mono_loss": health_mono_weight,
            "failure_loss": failure_weight,
            "asym_loss": asym_weight,
            "over_margin_loss": over_margin_weight,
        }
        self.asym_over_alpha = asym_over_alpha
        self.asym_under_beta = asym_under_beta
        self.asym_clip = asym_clip
        self.over_margin = over_margin
        self.late_stage_weight = late_stage_weight
        self.learnable_stage_prior = learnable_stage_prior
        self.stage_prior_reg_weight = stage_prior_reg_weight
        self.stage_prior_range = stage_prior_range
        initial_prior = torch.ones(num_stages, dtype=torch.float32)
        initial_prior[-1] = late_stage_weight
        initial_prior = initial_prior / initial_prior.mean().clamp_min(1e-12)
        if learnable_stage_prior:
            normalized_offset = ((initial_prior - 1.0) / stage_prior_range).clamp(-0.95, 0.95)
            initial_logits = 0.5 * torch.log((1.0 + normalized_offset) / (1.0 - normalized_offset))
            self.stage_logits = nn.Parameter(initial_logits)
        else:
            self.register_buffer("fixed_stage_prior", initial_prior)
        self.failure_threshold = failure_threshold
        self.physics_balancer = BRDRWeightBalancer(
            names=[
                "mono_loss",
                "smooth_loss",
                "rul_dynamics_loss",
                "sensor_trend_loss",
                "health_loss",
                "health_mono_loss",
                "failure_loss",
            ],
            beta_c=beta_c,
            beta_w=beta_w,
            min_weight=brdr_min_weight,
            max_weight=brdr_max_weight,
            weight_power=brdr_weight_power,
            warmup_steps=brdr_warmup_steps,
        )
        self.condition_stage_names = [
            f"condition_{condition_idx}_stage_{stage_idx}"
            for condition_idx in range(num_conditions)
            for stage_idx in range(num_stages)
        ]
        self.condition_stage_balancer = BRDRWeightBalancer(
            names=self.condition_stage_names,
            beta_c=beta_c,
            beta_w=beta_w,
            min_weight=brdr_min_weight,
            max_weight=brdr_max_weight,
            weight_power=brdr_weight_power,
            warmup_steps=brdr_warmup_steps,
        )
        self.last_data_objective: torch.Tensor | None = None
        self.last_physics_objective: torch.Tensor | None = None
        self.last_weighted_physics_objectives: dict[str, torch.Tensor] = {}

    def stage_prior(self, device: torch.device) -> torch.Tensor:
        if self.learnable_stage_prior:
            prior = 1.0 + self.stage_prior_range * torch.tanh(self.stage_logits)
            return prior / prior.mean().clamp_min(1e-12)
        return self.fixed_stage_prior.to(device)

    def forward(
        self,
        pred_seq: torch.Tensor,
        target_seq: torch.Tensor,
        condition_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        x_seq: torch.Tensor | None = None,
        health_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred_last = pred_seq[:, -1]
        target_last = target_seq[:, -1]
        condition_ids = condition_ids.long()
        stage_ids = stage_ids.long()

        sample_rul_sq = (pred_last - target_last).pow(2)
        batch_rul_loss = sample_rul_sq.mean()

        group_values = {}
        for condition_idx in range(self.num_conditions):
            for stage_idx in range(self.num_stages):
                name = f"condition_{condition_idx}_stage_{stage_idx}"
                mask = (condition_ids == condition_idx) & (stage_ids == stage_idx)
                if mask.any():
                    group_values[name] = sample_rul_sq[mask].mean()

        group_weights = self.condition_stage_balancer.update(
            group_values,
            active_names=set(group_values),
        )

        sample_weights = pred_last.new_ones(len(pred_last))
        for condition_idx in range(self.num_conditions):
            for stage_idx in range(self.num_stages):
                mask = (condition_ids == condition_idx) & (stage_ids == stage_idx)
                if mask.any():
                    name = f"condition_{condition_idx}_stage_{stage_idx}"
                    sample_weights[mask] = group_weights[name]

        stage_prior = self.stage_prior(pred_last.device)
        sample_weights = sample_weights * stage_prior[stage_ids]
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-12)

        rul_loss = (sample_weights * sample_rul_sq).mean()

        if x_seq is not None and self.sensor_trend_signs.dim() == 2:
            condition_signs = self.sensor_trend_signs[condition_ids]
        elif x_seq is not None:
            condition_signs = self.sensor_trend_signs.unsqueeze(0).expand(len(x_seq), -1)
        if x_seq is not None and self.sensor_trend_signs.abs().sum() > 0:
            health_score = (
                (x_seq - self.condition_sensor_baseline[condition_ids].to(x_seq.device).unsqueeze(1))
                * condition_signs.to(x_seq.device).unsqueeze(1)
            ).sum(dim=-1) / condition_signs.abs().sum(dim=-1).clamp_min(1.0).unsqueeze(1)
            health_low = self.health_low[condition_ids].to(x_seq.device).unsqueeze(1)
            health_high = self.health_high[condition_ids].to(x_seq.device).unsqueeze(1)
            health_target = 1.0 - (
                (health_score - health_low) / (health_high - health_low).clamp_min(1e-6)
            ).clamp(0.0, 1.0)
        else:
            health_target = (target_seq / self.max_rul).clamp(0.0, 1.0)

        stage_seq = torch.zeros_like(target_seq, dtype=torch.long)
        stage_seq[(target_seq <= self.middle_rul) & (target_seq > self.late_rul)] = 1
        stage_seq[target_seq <= self.late_rul] = 2
        learned_rul_rate = pred_seq.new_ones(pred_seq[:, 1:].shape)
        rul_step_residual = pred_seq[:, 1:] - pred_seq[:, :-1] + learned_rul_rate
        rul_dynamics_mask = target_seq[:, 1:] <= self.rul_dynamics_threshold
        sensor_stage_mask = (stage_seq[:, 1:] == stage_seq[:, :-1]) & (
            stage_seq[:, 1:] >= self.sensor_trend_min_stage
        )
        if x_seq is not None and self.sensor_trend_signs.abs().sum() > 0:
            sensor_delta = x_seq[:, 1:, :] - x_seq[:, :-1, :]
            if self.sensor_trend_signs.dim() == 2:
                trend_signs = self.sensor_trend_signs[condition_ids]
            else:
                trend_signs = self.sensor_trend_signs.unsqueeze(0).expand(len(x_seq), -1)
            sensor_degradation_delta = (
                sensor_delta * trend_signs.to(x_seq.device).unsqueeze(1)
            ).sum(dim=-1) / trend_signs.abs().sum(dim=-1).clamp_min(1.0).unsqueeze(1)
            sensor_trend_product = sensor_degradation_delta * (
                pred_seq[:, 1:] - pred_seq[:, :-1]
            )
        else:
            sensor_stage_mask = torch.zeros_like(sensor_stage_mask, dtype=torch.bool)
            sensor_trend_product = pred_last.new_zeros(sensor_stage_mask.shape)

        health_prediction = health_target.detach() if health_seq is None else health_seq
        mono_values = torch.relu(pred_seq[:, 1:] - pred_seq[:, :-1])
        smooth_values = ((pred_seq[:, 1:] - pred_seq[:, :-1]) + 1.0).pow(2)
        if self.weak_health_index and health_seq is not None:
            health_drop = health_prediction[:, :-1] - health_prediction[:, 1:]
            pairwise_values = F.relu(self.health_margin - health_drop)
            healthy_mask = target_seq[:, -1] >= self.middle_rul
            failure_mask = target_seq[:, -1] <= self.failure_threshold
            healthy_anchor = (
                F.relu(0.8 - health_prediction[healthy_mask, 0]).mean()
                if healthy_mask.any() else pred_last.new_tensor(0.0)
            )
            failure_anchor = (
                health_prediction[failure_mask, -1].pow(2).mean()
                if failure_mask.any() else pred_last.new_tensor(0.0)
            )
            hi_std = health_prediction.std(dim=1, unbiased=False)
            variance_penalty = F.relu(self.hi_min_std - hi_std).pow(2).mean()
            if healthy_mask.any() and failure_mask.any():
                endpoint_gap = F.relu(
                    self.hi_endpoint_gap
                    - (health_prediction[healthy_mask, 0].mean() - health_prediction[failure_mask, -1].mean())
                ).pow(2)
            else:
                endpoint_gap = pred_last.new_tensor(0.0)
            range_penalty = (
                F.relu(self.hi_range_eps - health_prediction).pow(2)
                + F.relu(health_prediction - (1.0 - self.hi_range_eps)).pow(2)
            ).mean()
            health_values = (
                pairwise_values.mean()
                + 0.5 * healthy_anchor
                + 0.5 * failure_anchor
                + variance_penalty
                + endpoint_gap
                + 0.1 * range_penalty
            )
        else:
            health_values = (health_prediction - health_target).pow(2).mean()
        health_mono_values = torch.relu(health_prediction[:, 1:] - health_prediction[:, :-1])
        component_losses = {
            "mono_loss": mono_values.mean(),
            "smooth_loss": smooth_values.mean(),
            "rul_dynamics_loss": (
                (
                    rul_step_residual.pow(2)
                )[rul_dynamics_mask].mean()
                if rul_dynamics_mask.any()
                else pred_last.new_tensor(0.0)
            ),
            "sensor_trend_loss": (
                F.huber_loss(
                    torch.relu(sensor_trend_product[sensor_stage_mask]),
                    torch.zeros_like(sensor_trend_product[sensor_stage_mask]),
                    delta=self.sensor_trend_huber_delta,
                    reduction="none",
                ).mean()
                if sensor_stage_mask.any()
                else pred_last.new_tensor(0.0)
            ),
            "health_loss": health_values,
            "health_mono_loss": health_mono_values.mean(),
        }
        near_failure = target_last <= self.failure_threshold
        if near_failure.any():
            component_losses["failure_loss"] = pred_last[near_failure].pow(2).mean()
        else:
            component_losses["failure_loss"] = pred_last.new_tensor(0.0)
        error = pred_last - target_last
        over_penalty = torch.exp((error / self.asym_over_alpha).clamp(max=self.asym_clip)) - 1.0
        under_penalty = torch.exp((-error / self.asym_under_beta).clamp(max=self.asym_clip)) - 1.0
        component_losses["asym_loss"] = torch.where(error >= 0, over_penalty, under_penalty).mean()
        component_losses["over_margin_loss"] = torch.relu(error - self.over_margin).pow(2).mean()

        active_physics_names = {"mono_loss", "smooth_loss"}
        if rul_dynamics_mask.any():
            active_physics_names.add("rul_dynamics_loss")
        if sensor_stage_mask.any():
            active_physics_names.add("sensor_trend_loss")
        active_physics_names.update({"health_loss", "health_mono_loss"})
        if near_failure.any():
            active_physics_names.add("failure_loss")
        physics_weights = self.physics_balancer.update(
            component_losses,
            active_names=active_physics_names,
        )

        data_objective = self.base_weights["rul_loss"] * rul_loss
        data_objective = data_objective + self.base_weights["asym_loss"] * component_losses["asym_loss"]
        data_objective = data_objective + self.base_weights["over_margin_loss"] * component_losses["over_margin_loss"]
        weighted_physics = {
            name: self.base_weights[name] * physics_weights.get(name, 1.0) * component_losses[name]
            for name in (
                "mono_loss",
                "smooth_loss",
                "rul_dynamics_loss",
                "sensor_trend_loss",
                "health_loss",
                "health_mono_loss",
                "failure_loss",
            )
            if self.base_weights[name] > 0
        }
        physics_objective = pred_last.new_tensor(0.0)
        for weighted_loss in weighted_physics.values():
            physics_objective = physics_objective + weighted_loss
        self.last_data_objective = data_objective
        self.last_physics_objective = physics_objective
        self.last_weighted_physics_objectives = weighted_physics
        # Preserve the historical summation order for an exact diagnostic-only baseline.
        total = self.base_weights["rul_loss"] * rul_loss
        for name, component_loss in component_losses.items():
            total = total + self.base_weights[name] * physics_weights.get(name, 1.0) * component_loss
        stage_prior_reg = (stage_prior - 1.0).pow(2).mean()
        if self.learnable_stage_prior and self.stage_prior_reg_weight > 0:
            total = total + self.stage_prior_reg_weight * stage_prior_reg

        parts = {
            "total_loss": float(total.detach().cpu()),
            "rul_loss": float(rul_loss.detach().cpu()),
            "raw_rul_loss": float(batch_rul_loss.detach().cpu()),
            "mono_loss": float(component_losses["mono_loss"].detach().cpu()),
            "smooth_loss": float(component_losses["smooth_loss"].detach().cpu()),
            "rul_dynamics_loss": float(component_losses["rul_dynamics_loss"].detach().cpu()),
            "sensor_trend_loss": float(component_losses["sensor_trend_loss"].detach().cpu()),
            "health_loss": float(component_losses["health_loss"].detach().cpu()),
            "health_mono_loss": float(component_losses["health_mono_loss"].detach().cpu()),
            "failure_loss": float(component_losses["failure_loss"].detach().cpu()),
            "asym_loss": float(component_losses["asym_loss"].detach().cpu()),
            "over_margin_loss": float(component_losses["over_margin_loss"].detach().cpu()),
            "w_rul_loss": 1.0,
            "w_mono_loss": float(physics_weights["mono_loss"]),
            "w_smooth_loss": float(physics_weights["smooth_loss"]),
            "w_rul_dynamics_loss": float(physics_weights["rul_dynamics_loss"]),
            "w_sensor_trend_loss": float(physics_weights["sensor_trend_loss"]),
            "w_health_loss": float(physics_weights["health_loss"]),
            "w_health_mono_loss": float(physics_weights["health_mono_loss"]),
            "w_failure_loss": float(physics_weights["failure_loss"]),
            "w_asym_loss": 1.0,
            "w_over_margin_loss": 1.0,
            "stage_prior_reg": float(stage_prior_reg.detach().cpu()),
        }
        for condition_idx in range(self.num_conditions):
            stage_values = []
            for stage_idx in range(self.num_stages):
                name = f"condition_{condition_idx}_stage_{stage_idx}"
                parts[f"w_{name}"] = float(group_weights[name])
                stage_values.append(group_weights[name])
            parts[f"w_condition_{condition_idx}"] = float(sum(stage_values) / len(stage_values))
        for stage_idx in range(self.num_stages):
            values = [
                group_weights[f"condition_{condition_idx}_stage_{stage_idx}"]
                for condition_idx in range(self.num_conditions)
            ]
            parts[f"w_stage_{stage_idx}"] = float(sum(values) / len(values))
            parts[f"prior_stage_{stage_idx}"] = float(stage_prior[stage_idx].detach().cpu())
        return total, parts
