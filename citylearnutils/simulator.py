import numpy as np
import numba as nb


@nb.njit()
def _simulate(action: float, soc: float, capacity: float, nominal_power: float, capacity_power_curve: np.ndarray,
              power_efficiency_curve: np.ndarray, efficiency_scaling: float, loss_coefficient: float,
              capacity_loss_coefficient: float, init_capacity: float):
    """Maps a (state, action) pair into the next state"""

    # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
    soc_normalized = soc / capacity

    # Calculating the maximum power rate at which the battery can be charged or discharged
    idx = max(0, np.argmax(soc_normalized <= capacity_power_curve[0]) - 1)

    intercept = capacity_power_curve[1][idx]

    slope = (capacity_power_curve[1][idx + 1] - capacity_power_curve[1][idx]) / \
            (capacity_power_curve[0][idx + 1] - capacity_power_curve[0][idx])

    max_power = intercept + slope * (soc_normalized - capacity_power_curve[0][idx])
    max_power *= nominal_power

    # calculate the energy of the action
    if action >= 0:
        energy = min(action * capacity, max_power)
    else:
        energy = max(-max_power, action * capacity)

    # Calculating the maximum power rate at which the battery can be charged or discharged
    energy_normalized = np.abs(energy) / nominal_power
    idx = max(0, np.argmax(energy_normalized <= power_efficiency_curve[0]) - 1)

    intercept = power_efficiency_curve[1][idx]

    slope = (power_efficiency_curve[1][idx + 1] - power_efficiency_curve[1][idx]) / \
            (power_efficiency_curve[0][idx + 1] - power_efficiency_curve[0][idx])

    efficiency = intercept + slope * (energy_normalized - power_efficiency_curve[0][idx])
    efficiency = efficiency ** efficiency_scaling

    # update state of charge
    if energy >= 0:
        next_soc = min(soc + energy * efficiency, capacity)
    else:
        next_soc = max(0.0, soc + energy / efficiency)

    # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge
    energy_balance = next_soc - soc * (1 - loss_coefficient)
    energy_balance *= (1 / efficiency if energy_balance >= 0 else efficiency)
    next_capacity = capacity - capacity_loss_coefficient * init_capacity * abs(energy_balance) / (2 * capacity)

    # calculate batter consumption
    soc_diff = next_soc - soc
    battery_energy = soc_diff * (1 / efficiency if soc_diff > 0 else efficiency)

    return next_soc, next_capacity, efficiency, battery_energy


class BatterySimulator:
    def __init__(self, init_capacity=6.4, nominal_power=5.0, loss_coefficient=0.0, efficiency_scaling=0.5,
                 capacity_loss_coefficient=1e-5):
        r"""Initialize `Battery`.

        Parameters
        ----------
        init_capacity : float
        nominal_power: float
        loss_coefficient : float
        efficiency_scaling : float
        capacity_loss_coefficient : float

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super classes.
        """

        self.init_capacity = init_capacity
        self.nominal_power = nominal_power
        self.loss_coefficient = loss_coefficient
        self.efficiency_scaling = efficiency_scaling
        self.capacity_loss_coefficient = capacity_loss_coefficient
        self.capacity_power_curve = np.array([[0., 0.8, 1.], [1., 1., 0.2]])
        self.power_efficiency_curve = np.array([[0., 0.3, 0.7, 0.8, 1.], [0.83, 0.83, 0.9, 0.9, 0.85]])

    def fast_simulate(self, action, current_soc, current_capacity):
        """Simulates charging. Doesn't update input or internal any state.

        Parameters
        ----------
        action : float
        current_soc : float
        current_capacity : float
        """
        return _simulate(
            action, current_soc, current_capacity,
            nominal_power=self.nominal_power,
            capacity_power_curve=self.capacity_power_curve,
            power_efficiency_curve=self.power_efficiency_curve,
            efficiency_scaling=self.efficiency_scaling,
            loss_coefficient=self.loss_coefficient,
            capacity_loss_coefficient=self.capacity_loss_coefficient,
            init_capacity=self.init_capacity
        )

    def simulate(self, action, current_soc, current_capacity):
        """Simulates charging. Doesn't update input or internal any state.

        Parameters
        ----------
        action : float
        current_soc : float
        current_capacity : float
        """

        if action >= 0:
            energy = min(action * current_capacity, self._get_max_power(current_soc, current_capacity))
        else:
            energy = max(-self._get_max_power(current_soc, current_capacity), action * current_capacity)

        # update efficiency
        current_efficiency = self._get_current_efficiency(energy)

        # update state of charge
        if energy >= 0.0:
            next_soc = min(current_soc + energy * current_efficiency, current_capacity)
        else:
            next_soc = max(0.0, current_soc + energy / current_efficiency)

        # update capacity
        next_capacity = current_capacity - self._get_degradation(
            current_soc, next_soc, current_capacity, current_efficiency
        )

        return next_soc, next_capacity, current_efficiency, energy

    def _get_max_power(self, soc, capacity) -> float:
        r"""Get maximum input power while considering `capacity_power_curve` limitations if defined otherwise, returns
        `nominal_power`.

        Returns
        -------
        max_power : float
            Maximum amount of power that the storage unit can use to charge [kW].
        """

        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        soc_normalized = soc / capacity

        # Calculating the maximum power rate at which the battery can be charged or discharged
        idx = max(0, np.argmax(soc_normalized <= self.capacity_power_curve[0]) - 1)

        intercept = self.capacity_power_curve[1][idx]

        slope = (self.capacity_power_curve[1][idx + 1] - self.capacity_power_curve[1][idx]) / \
                (self.capacity_power_curve[0][idx + 1] - self.capacity_power_curve[0][idx])

        max_power = intercept + slope * (soc_normalized - self.capacity_power_curve[0][idx])

        max_power *= self.nominal_power

        return max_power

    def _get_current_efficiency(self, energy: float) -> float:
        r"""Get technical efficiency while considering `power_efficiency_curve` limitations if defined otherwise,
        returns `efficiency`.

        Returns
        -------
        efficiency : float
            Technical efficiency.
        """

        # Calculating the maximum power rate at which the battery can be charged or discharged
        energy_normalized = np.abs(energy) / self.nominal_power
        idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)

        intercept = self.power_efficiency_curve[1][idx]

        slope = (self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx]) / \
                (self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])

        efficiency = intercept + slope * (energy_normalized - self.power_efficiency_curve[0][idx])

        efficiency = efficiency ** self.efficiency_scaling

        return efficiency

    def _get_degradation(self, current_soc, next_soc, current_capacity, current_efficiency) -> float:
        r"""Get amount of capacity degradation.

        Returns
        -------
        current_soc : float
        soc : float
        current_capacity : float
        efficiency : float
        """

        # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge
        energy_balance = next_soc - current_soc * (1 - self.loss_coefficient)
        energy_balance *= (1 / current_efficiency if energy_balance >= 0 else current_efficiency)
        return self.capacity_loss_coefficient * self.init_capacity * abs(energy_balance) / (2 * current_capacity)

