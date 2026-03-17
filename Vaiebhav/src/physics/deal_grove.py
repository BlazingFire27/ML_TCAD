# src/physics/deal_grove.py
import numpy as np

try:
    from scipy.optimize import least_squares
except Exception as e:
    least_squares = None  # fit() will raise an instructive error if SciPy is missing

# Boltzmann constant in eV/K (for Arrhenius)
k_B_eV_per_K = 8.617e-5


class DealGrove:
    """
    Deal–Grove utility:
      - compute partial pressure (supports per-point pressure OR fixed default)
      - DG predictor given kinetics constants (B0, EB, BA0, EBA)
      - sub-grid linear interpolation of interface from discrete X, Y profile
      - extraction of observed thickness from profile
      - fit kinetics constants to observed thickness data (requires scipy)

    All I/O uses numpy arrays / scalars. Time expected in minutes (per your dataset).
    Temperature expected in degree Celsius (per your dataset).
    Flows are used as-is to compute mole/flow fraction; units cancel.
    Pressure can be passed per-point or use the default total_pressure_atm.
    """

    def __init__(self, total_pressure_atm: float = 0.44, reactive_threshold_logY: float = 2.0):
        self.P_total_atm = float(total_pressure_atm)
        self.reactive_threshold = float(reactive_threshold_logY)

    # -------------------------
    # Partial pressure (user requested fixed-total-pressure approach)
    # -------------------------
    def compute_partial_pressure(self, o2_flow, n2_flow, pressure=None):
        """
        Compute pO2 (in atm) from O2 and N2 flows and total pressure.
        
        Args:
            o2_flow: O2 flow rate (scalar or array)
            n2_flow: N2 flow rate (scalar or array)
            pressure: Total pressure in atm (scalar or array). If None, uses self.P_total_atm.
        """
        o2 = np.asarray(o2_flow, dtype=float)
        n2 = np.asarray(n2_flow, dtype=float)
        frac = o2 / (o2 + n2)
        if pressure is not None:
            P = np.asarray(pressure, dtype=float)
        else:
            P = self.P_total_atm
        return frac * P

    # -------------------------
    # Deal-Grove predictor (given kinetics constants)
    # -------------------------
    def predict_with_constants(self, temperature_C, time_min, o2_flow, n2_flow,
                               B0, EB, BA0, EBA, tau_hr: float = 0.0,
                               pressure=None):
        """
        Predict oxide thickness X_ox (same units as the B0/BA0 conventions).
        - temperature_C : scalar or array (°C)
        - time_min : scalar or array (minutes)
        - o2_flow, n2_flow : scalars or arrays
        - B0, EB, BA0, EBA : kinetics constants (B0, BA0 > 0)
        - tau_hr: time-shift in hours (default 0.0)
        - pressure: total pressure in atm (scalar or array). If None, uses self.P_total_atm.

        Note: Because kinetics constants may be in various units in literature,
        the units of X returned depend on the units used during fit. We intentionally
        do not hard-code unit conversions for B0/BA0 — treat B0/BA0 consistent with observed thickness units.
        """
        # broadcast inputs
        temperature_C = np.asarray(temperature_C, dtype=float)
        time_min = np.asarray(time_min, dtype=float)
        o2_flow = np.asarray(o2_flow, dtype=float)
        n2_flow = np.asarray(n2_flow, dtype=float)

        # conversions
        T_K = temperature_C + 273.15
        t_hr = (time_min / 60.0) + float(tau_hr)

        pO2 = self.compute_partial_pressure(o2_flow, n2_flow, pressure=pressure)

        # Arrhenius forms
        B = B0 * np.exp(-EB / (k_B_eV_per_K * T_K)) * pO2
        BA = BA0 * np.exp(-EBA / (k_B_eV_per_K * T_K)) * pO2

        # Avoid divide-by-zero in degenerate BA (user should fit reasonable values)
        A = B / BA

        # Solve quadratic X^2 + A X - B t = 0  => positive root
        disc = A**2 + 4.0 * B * t_hr
        # numerical safety
        disc = np.maximum(disc, 0.0)
        X = (-A + np.sqrt(disc)) / 2.0
        return X

    # -------------------------
    # Sub-grid linear interpolation of interface
    # -------------------------
    def interpolate_interface(self, X_array, Y_array_log10=None, raw_Y=None):
        """
        Given a 1D ordered spatial array X_array and either log10(Y) (Y_array_log10)
        or raw Y concentration (raw_Y), compute a sub-grid interface x_interface by
        linearly interpolating between last 'reactive' point and first 'bulk' point.

        Returns x_interface (float).

        Raises ValueError if no reactive point found (i.e., no point with logY > reactive_threshold).
        """
        X = np.asarray(X_array, dtype=float)
        if Y_array_log10 is None:
            if raw_Y is None:
                raise ValueError("Provide either Y_array_log10 or raw_Y (concentrations).")
            logY = np.log10(np.clip(np.asarray(raw_Y, dtype=float), 1e-12, None))
        else:
            logY = np.asarray(Y_array_log10, dtype=float)

        mask = logY > self.reactive_threshold
        if not np.any(mask):
            raise ValueError("No reactive (oxide) point found in profile with given threshold.")

        # last oxide index
        i1 = np.where(mask)[0].max()
        i2 = i1 + 1
        # if i2 is out of bounds, we cannot interpolate — place interface at X[i1]
        if i2 >= len(X):
            return float(X[i1])

        X1, X2 = X[i1], X[i2]
        Y1, Y2 = logY[i1], logY[i2]

        # if the two logY values are equal (rare), return X1
        if np.isclose(Y2, Y1):
            return float(X1)

        # linear interpolation to threshold
        x_interface = X1 + (self.reactive_threshold - Y1) * (X2 - X1) / (Y2 - Y1)
        return float(x_interface)

    # -------------------------
    # Extract observed thickness from a single profile
    # -------------------------
    def extract_thickness_from_profile(self, X_array, raw_Y):
        """
        Given X_array (ordered) and raw Y (concentration), compute interface by interpolation
        and return thickness = interface_x - surface_x (surface assumed min(X_array)).
        """
        logY = np.log10(np.clip(np.asarray(raw_Y, dtype=float), 1e-12, None))
        interface_x = self.interpolate_interface(X_array, Y_array_log10=logY)
        surface_x = float(np.min(X_array))
        thickness = interface_x - surface_x
        return float(thickness)

    # -------------------------
    # Fit kinetics constants to observed thicknesses
    # -------------------------
    def fit(self,
            temperature_C_arr,
            time_min_arr,
            o2_flow_arr,
            n2_flow_arr,
            thickness_obs_arr,
            *,
            pressure_arr=None,
            x0=None,
            bounds=None,
            tau_hr: float = 0.0,
            verbose: bool = False):
        """
        Fit kinetics constants [logB0, EB, logBA0, EBA] to observed thicknesses via nonlinear least squares.
        We parameterize prefactors in log-space to keep them positive:
            B0 = exp(logB0), BA0 = exp(logBA0)

        Inputs: all arrays must be same shape (N,)
        - temperature_C_arr, time_min_arr, o2_flow_arr, n2_flow_arr : process inputs
        - thickness_obs_arr : observed thickness (same units as the resulting prediction)

        Optional:
        - pressure_arr: total pressure array (same shape as inputs). If None, uses self.P_total_atm.
        - x0 : initial guess vector [logB0, EB, logBA0, EBA] (if None, the code supplies a default guess)
        - bounds: pair (lower, upper) bounds for least_squares
        - tau_hr: optional time shift (hours) to add to time during prediction
        """
        if least_squares is None:
            raise RuntimeError("scipy.optimize.least_squares required for fit(). Install scipy and retry.")

        # pack arrays
        T = np.asarray(temperature_C_arr, dtype=float)
        tmin = np.asarray(time_min_arr, dtype=float)
        o2 = np.asarray(o2_flow_arr, dtype=float)
        n2 = np.asarray(n2_flow_arr, dtype=float)
        pres = np.asarray(pressure_arr, dtype=float) if pressure_arr is not None else None
        y_obs = np.asarray(thickness_obs_arr, dtype=float)

        # default initial guess if not provided (log prefactors + energies)
        if x0 is None:
            # small, generic guesses; these are only starting points for optimizer (no physics assumptions kept)
            # Note: we use log-space for prefactors to ensure positivity
            x0 = np.array([np.log(1e2), 1.0, np.log(1e5), 1.5], dtype=float)

        # default bounds (wide)
        if bounds is None:
            lower = np.array([np.log(1e-12), 0.01, np.log(1e-12), 0.01], dtype=float)
            upper = np.array([np.log(1e12), 5.0, np.log(1e12), 5.0], dtype=float)
            bounds = (lower, upper)

        def residuals(p):
            # p = [logB0, EB, logBA0, EBA]
            logB0, EB, logBA0, EBA = p
            B0 = np.exp(logB0)
            BA0 = np.exp(logBA0)
            y_pred = self.predict_with_constants(T, tmin, o2, n2, B0=B0, EB=EB, BA0=BA0, EBA=EBA,
                                                  tau_hr=tau_hr, pressure=pres)
            return (y_pred - y_obs).ravel()

        res = least_squares(residuals, x0, bounds=bounds, verbose=2 if verbose else 0, xtol=1e-10, ftol=1e-10)

        # unpack fitted parameters
        logB0_fit, EB_fit, logBA0_fit, EBA_fit = res.x
        fitted = {
            "B0": float(np.exp(logB0_fit)),
            "EB": float(EB_fit),
            "BA0": float(np.exp(logBA0_fit)),
            "EBA": float(EBA_fit),
            "success": bool(res.success),
            "cost": float(res.cost),
            "message": res.message
        }
        return fitted

    # -------------------------
    # Utility: predict using fitted dictionary
    # -------------------------
    def predict_with_fitted(self, temperature_C, time_min, o2_flow, n2_flow, fitted_params,
                             tau_hr: float = 0.0, pressure=None):
        """
        Convenience wrapper: given 'fitted_params' dict (keys B0, EB, BA0, EBA), predict thickness.
        pressure: optional per-point pressure array. If None, uses self.P_total_atm.
        """
        return self.predict_with_constants(temperature_C, time_min, o2_flow, n2_flow,
                                           B0=fitted_params["B0"],
                                           EB=fitted_params["EB"],
                                           BA0=fitted_params["BA0"],
                                           EBA=fitted_params["EBA"],
                                           tau_hr=tau_hr,
                                           pressure=pressure)