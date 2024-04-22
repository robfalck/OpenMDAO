import copy

from collections.abc import Callable, Iterable
from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.component import Component
from openmdao.components.mux_comp import MuxComp

class VectorizedSystem(Group):

    def initialize(self):
        self.options.declare(
            'vec_size', types=int,
            desc='The number of executions of the system to be vectorized.'
        )

        self.options.declare(
            'vec_inputs', types=list, allow_none=False,
            desc='The inputs to the system which are to be vectorized. If additional metadata is specified, such '
                 ' as shape or units, then the input should be specified as a two-element tuple where the first '
                 ' element is the string variable name as the second is a dictionary of metadata.'
        )

        self.options.declare(
            'scalar_inputs', types=list, default=[],
            desc='The inputs to the system which are to be vectorized. If additional metadata is specified, such '
                 ' as shape or units, then the input should be specified as a two-element tuple where the first '
                 ' element is the string variable name as the second is a dictionary of metadata.'
        )

        self.options.declare(
            'vec_outputs', types=list, allow_none=False,
            desc='The outputs of the system which are to be vectorized. If additional metadata is specified, such '
                 ' as shape or units, then the input should be specified as a two-element tuple where the first '
                 ' element is the string variable name as the second is a dictionary of metadata.'
        )

        self.options.declare(
            'system', types=(Group, Component),
            desc='The class of the system to be vectorized.'
        )

        self.options.declare(
            'vec_fmt', types=str, default='{}_vec',
            desc='The format string for the vectorized variables.'
        )

    def setup(self):
        n = self.options['vec_size']
        system = self.options['system']
        vec_inputs = self.options['vec_inputs']
        vec_outputs = self.options['vec_outputs']
        scalar_inputs = self.options['scalar_inputs']
        vec_fmt = self.options['vec_fmt']

        parallel = self.add_subsystem('parallel', ParallelGroup(), promotes_inputs=['*'])
        self.add_subsystem('mux', MuxComp(vec_size=n), promotes_outputs=['*'])

        for idx in range(n):
            parallel.add_subsystem(name=f'sys_{idx}', subsys=copy.deepcopy(system))

    def configure(self):
        n = self.options['vec_size']
        vec_fmt = self.options['vec_fmt']
        scalar_inputs = self.options['scalar_inputs']

        sys = self._get_subsystem('parallel.sys_0')
        parallel = self._get_subsystem('parallel')
        mux = self._get_subsystem('mux')

        inputs = {v['prom_name']: v for v in sys.get_io_metadata(iotypes=('input',), get_remote=True).values()}
        outputs = {v['prom_name']: v for v in sys.get_io_metadata(iotypes=('output',), get_remote=True).values()}

        vec_inputs = {s: {} for s in self.options['vec_inputs'] if isinstance(s, str)}
        vec_inputs.update({s[0]: s[1] for s in self.options['vec_inputs'] if not isinstance(s, str)})
        vec_outputs = {s: {} for s in self.options['vec_outputs'] if isinstance(s, str)}
        vec_outputs.update({s[0]: s[1] for s in self.options['vec_outputs'] if not isinstance(s, str)})

        for idx in range(n):
            # Connect the current index for each input
            for inp, meta in inputs.items():
                if inp in vec_inputs:
                    shape = meta['shape']
                    src_shape = (n,) + shape
                    parallel.promotes(f'sys_{idx}', inputs=[(inp, vec_fmt.format(inp))], src_indices=[idx], src_shape=src_shape)
                else:
                    parallel.promotes(f'sys_{idx}', inputs=[inp])

            for oup, user_meta in vec_outputs.items():
                self.connect(f'parallel.sys_{idx}.{oup}', f'mux.{oup}_{idx}')

            # All inputs not listed in vec_inputs are assumed to be scalar.
            parallel.promotes(f'sys_{idx}', [inp for inp in inputs if inp not in vec_inputs])

        for oup, user_meta in vec_outputs.items():
            meta = user_meta if user_meta else outputs[oup]
            mux.add_var(oup, shape=meta['shape'], units=meta['units'], axis=1)
            self.promotes('mux', outputs=[(oup, vec_fmt.format(oup))])


if __name__ == '__main__':
    import openmdao.api as om
    import numpy as np
    prob = om.Problem()
    prob.model.add_subsystem(
        'vec_sys', VectorizedSystem(
            vec_size=4, vec_inputs=['x'], vec_outputs=['y'],
            system=om.ExecComp(['y=-1.0*x**2'], x={'units': 'm'}, y={'units': 'm**2'})
        ),
        promotes_inputs=['*'], promotes_outputs=['*']
    )
    prob.setup()
    prob.set_val('x_vec', np.array([1.25, 1.5, 1.75, 2.0]))
    prob.run_model()
    prob.model.list_inputs()
    prob.model.list_outputs(units=True)
