import numpy as np

import openmdao.api as om


class TanhRampComp(om.ExplicitComponent):
    """ Differentiable transition from one steady state condition to another using a hyperbolic tangent.
    The hyperbolic tangent activation function is:
    r_1(t) = np.tanh(t)
    This has a response centered about t=0, and a value of (-0.996, 0.996) at (-np.pi, np.pi) (thus the duration
    is roughly 2*np.pi.
    This implementation allows that response to be shifted and stretch such that the user can specify:
    1. the initial steady state value before the ramp.
    2. the final steady state value after the ramp.
    3. the starting time of the ramp (where we assume the nominal start time is -np.pi).
    4. the duration of the ramp (where we assume the nominal duration of the ramp is 2 * np.pi.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ramps = {}

    def add_response(self, name, units=None, shape=(1,), initial_val=0.0,
                 final_val=1.0, t_init_val=0.0, t_duration_val=1.0):
        """
        Add a tanh ramp function with the given output name to the component.
        Parameters
        ----------
        output_name : str
            The name of the output response variable.
        output_units : str or None
            Units of the response variable.
        shape : tuple
            Shape of the variable subject to the ramp at each node.
        initial_val : float
            The initial value that the ramp asymptotically approaches backward in time.
        final_val : float
            The final value that the ramp asymptotically approaches forward in time.
        t_init_val : float
            The default value for the time at which the ramp is initiated. Note that the value asymptotically
            departs the initial value and so it nearly but not exactly the initial value at this point.
        t_duration_val : float
            The default value for the time after t_init_val at which the ramp is approximately equal to the desired
            final value. Again, the hyperbolic tangent function will never exactly equal the final value but it is
            relatively flat after this duration is expired.
        """
        self._ramps[output_name] = {'shape': shape,
                                    'units': output_units,
                                    'initial_val_name': f"{output_name}:initial_val",
                                    'initial_val': initial_val,
                                    'final_val_name': f"{output_name}:final_val",
                                    'final_val': final_val,
                                    't_init_name': f"{output_name}:t_init",
                                    't_init_val': t_init_val,
                                    't_duration_name': f"{output_name}:t_duration",
                                    't_duration_val': t_duration_val}

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("time_units", types=str, default='s', allow_none=True)

    def setup(self):
        nn = self.options["num_nodes"]
        ar = np.arange(nn, dtype=int)

        self.add_input("time", units=self.options["time_units"], shape=(nn,))

        for output_name, options in self._ramps.items():
            size = np.prod(options['shape'], dtype=int)
            cs = np.tile(np.arange(size, dtype=int), nn)

            self.add_output(output_name, shape=(nn,) +
                            options['shape'], units=options['units'])

            self.add_input(options['initial_val_name'], val=options['initial_val']
                           * np.ones(options['shape']), units=options['units'])
            self.add_input(options['final_val_name'], val=options['final_val']
                           * np.ones(options['shape']), units=options['units'])
            self.add_input(
                options['t_init_name'], val=options['t_init_val'], units=self.options['time_units'])
            self.add_input(
                options['t_duration_name'], val=options['t_duration_val'], units=self.options['time_units'])

            self.declare_partials(
                of=output_name, wrt=f"{output_name}:initial_val", rows=ar, cols=cs)
            self.declare_partials(
                of=output_name, wrt=f"{output_name}:final_val", rows=ar, cols=cs)

            self.declare_partials(
                of=output_name, wrt=f"{output_name}:t_init", rows=ar, cols=np.zeros(nn, dtype=int))
            self.declare_partials(of=output_name, wrt=f"{output_name}:t_duration",
                                  rows=ar, cols=np.zeros(nn, dtype=int))
            self.declare_partials(of=output_name, wrt="time", rows=ar, cols=ar)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        t = inputs["time"]

        for name, options in self._ramps.items():
            initial_val = inputs[options["initial_val_name"]]
            final_val = inputs[options["final_val_name"]]
            t_init = inputs[options["t_init_name"]]
            t_duration = inputs[options["t_duration_name"]]

            dval = final_val - initial_val
            tanh_term = np.tanh(2 * np.pi * (t - t_init) / t_duration - np.pi)
            outputs[name] = 0.5 * dval * (1+tanh_term) + initial_val

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        t = inputs["time"]

        for name, options in self._ramps.items():
            initial_val = inputs[options["initial_val_name"]]
            final_val = inputs[options["final_val_name"]]
            t_init = inputs[options["t_init_name"]]
            t_duration = inputs[options["t_duration_name"]]

            dval = final_val - initial_val
            dvald2 = 0.5 * dval

            tanh_term = np.tanh(2 * np.pi * (t - t_init) / t_duration - np.pi)
            omtanh2 = 1 - tanh_term ** 2

            dtanh_term_dt = omtanh2 * 2 * np.pi / t_duration
            dtanh_term_dtinit = -dtanh_term_dt

            dtanh_term_dtduration = omtanh2 * \
                (-2 * np.pi * (t - t_init) / t_duration ** 2)

            partials[name, "time"] = dvald2 * dtanh_term_dt
            partials[name, options["t_init_name"]] = dvald2 * dtanh_term_dtinit
            partials[name, options["t_duration_name"]] = dvald2 * dtanh_term_dtduration
            partials[name, options["initial_val_name"]] = -0.5 * (1+tanh_term) + 1.0
            partials[name, options["final_val_name"]] = 0.5 * (1+tanh_term)
