import numpy as np

import openmdao.api as om


class tanh_act():

    def __call__(self, *args, **kwargs):

def tanh_act(x, mu=1, z=0, a=-1, b=1):
    """
    A function which provides a differentiable activation function based on the hyperbolic tangent.

    Parameters
    ----------
    x : float or np.array
        The input at which the value of the activation function is to be computed.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function. As this value approaches zero
        the response approaches that of a step function.
    z : float
        The value of the independent variable about which the activation response is centered.
    a : float
        The initial value that the input asymptotically approaches negative infinity.
    b : float
        The final value that the input asymptotically approaches positive infinity.

    Returns
    -------
        The value of the activation response at the given input.
    """
    dy = b - a
    tanh_term = np.tanh((x - z) / mu)
    return 0.5 * dy * (1 + tanh_term) + a


def dtanh_act(x, mu=1.0, z=0.0, a=-1.0, b=1.0):
    """
    A function which provides a differentiable activation function based on the hyperbolic tangent.

    Parameters
    ----------
    x : float or np.array
        The input at which the value of the activation function is to be computed.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function. As this value approaches zero
        the response approaches that of a step function.
    z : float
        The value of the independent variable about which the activation response is centered.
    a : float
        The initial value that the input asymptotically approaches negative infinity.
    b : float
        The final value that the input asymptotically approaches positive infinity.

    Returns
    -------
    dict
        A dictionary which contains the partial derivatives of the tanh activation function wrt inputs, stored in the
        keys 'x', 'mu', 'z', 'a', 'b'.
    """
    dy = b - a
    xmz = x - z
    tanh_term = np.tanh(xmz / mu)
    partials = {'x': (0.5 * dy) / (mu * np.cosh(xmz / mu)**2),
                'mu': (-0.5 * dy * xmz) / (mu**2 * np.cosh(xmz / mu)**2),
                'z': (-0.5 * dy) / (mu * np.cosh(xmz / mu)**2),
                'a': 0.5 * (1 - tanh_term),
                'b': 0.5 * (1 + tanh_term)}

    return partials


class TanhActComp(om.ExplicitComponent):
    """ Differentiable transition from one steady state condition to another using a hyperbolic tangent.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._response_meta = {}

    def add_response(self, name, units=None, input_name='x', input_units=None, shape=(1,), a=-1.0, b=1.0, mu=0.1, z=0.0):
        """
        Add a tanh ramp function with the given output name to the component.
        Parameters
        ----------
        name : str
            The name of the output response variable.
        units : str or None
            Units of the response variable.
        input_name : str
            The name of the input variable used in the output of this response.
        input_units : str or None
            Units on the input associated with the response.
        shape : tuple
            Shape of the variable subject to the ramp at each node.
        a : float
            The initial value that the input asymptotically approaches negative infinity.
        b : float
            The final value that the input asymptotically approaches positive infinity.
        mu : float
            A shaping parameter that impacts the "abruptness" of the activation response. As mu approaches
            negative infinity, the behavior of the activation approaches a step function.
        z : float
            The value of the input at which the tanh response is centered.
        """
        self.response_meta[name] = {'units': units,
                                    'input_name': input_name,
                                    'input_units': input_units,
                                    'shape': shape,
                                    'a': a,
                                    'b': b,
                                    'mu': mu,
                                    'z': z }

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("time_units", types=str, default='s', allow_none=True)

    def setup(self):
        nn = self.options["num_nodes"]
        ar = np.arange(nn, dtype=int)

        self.add_input("time", units=self.options["time_units"], shape=(nn,))

        input_names = set()

        for name, options in self._response_meta.items():
            shape = options['shape']
            size = np.prod(shape, dtype=int)
            cs = np.tile(np.arange(size, dtype=int), nn)

            self.add_output(name, shape=(nn,) + shape, units=options['units'])

            if options['input_name'] not in input_names:
                self.add_input(options['input_name'], val=np.ones(shape), units=options['input_units'])

            self.add_input(f'name:{}', val=options['initial_val']
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


if __name__ == '__main__':
    print(dtanh_act(10.0, mu=0.01, z=10.1))