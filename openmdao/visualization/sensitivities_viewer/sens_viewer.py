import numpy as np
from openmdao.drivers.optimization_driver import OptimizationDriver

# Diverging colormaps using either Purple-Green or Blue-Red
# Use _make_diverging_colormap to generate otheres
# These are built using seaborn
PuGr_256 = ['#7865db', '#7966db', '#7a67db', '#7b68db', '#7c69dc', '#7d6bdc', '#7e6cdc', '#7f6ddc',
            '#806edc', '#816fdc', '#8170dd', '#8271dd', '#8372dd', '#8473dd', '#8574dd', '#8676de',
            '#8777de', '#8878de', '#8979de', '#8a7ade', '#8b7bde', '#8c7cdf', '#8d7ddf', '#8e7edf',
            '#8f7fdf', '#9081df', '#9182e0', '#9283e0', '#9384e0', '#9485e0', '#9486e0', '#9587e1',
            '#9688e1', '#9789e1', '#988ae1', '#998be1', '#9a8de1', '#9b8ee2', '#9c8fe2', '#9d90e2',
            '#9e91e2', '#9f92e2', '#a093e3', '#a194e3', '#a295e3', '#a396e3', '#a498e3', '#a599e3',
            '#a69ae4', '#a79be4', '#a79ce4', '#a89de4', '#a99ee4', '#aa9fe5', '#aba0e5', '#aca1e5',
            '#ada3e5', '#aea4e5', '#afa5e5', '#b0a6e6', '#b1a7e6', '#b2a8e6', '#b3a9e6', '#b4aae6',
            '#b5ace7', '#b6ade7', '#b7aee7', '#b8afe7', '#b9b0e7', '#bab1e8', '#bbb2e8', '#bcb4e8',
            '#bdb5e8', '#beb6e8', '#bfb7e9', '#c0b8e9', '#c1b9e9', '#c2bae9', '#c3bbe9', '#c3bce9',
            '#c4bdea', '#c5beea', '#c6c0ea', '#c7c1ea', '#c8c2ea', '#c9c3eb', '#cac4eb', '#cbc5eb',
            '#ccc6eb', '#cdc7eb', '#cec8eb', '#cfc9ec', '#d0cbec', '#d1ccec', '#d2cdec', '#d3ceec',
            '#d4cfed', '#d5d0ed', '#d6d1ed', '#d6d2ed', '#d7d3ed', '#d8d4ee', '#d9d6ee', '#dad7ee',
            '#dbd8ee', '#dcd9ee', '#dddaee', '#dedbef', '#dfdcef', '#e0ddef', '#e1deef', '#e2dfef',
            '#e3e1f0', '#e4e2f0', '#e5e3f0', '#e6e4f0', '#e7e5f0', '#e8e6f0', '#e9e7f1', '#e9e8f1',
            '#eae9f1', '#ebeaf1', '#ecebf1', '#ededf2', '#eeeef2', '#efeff2', '#f0f0f2', '#f1f1f2',
            '#eff2ef', '#ecf2ea', '#ebf1e9', '#e9f0e7', '#e8efe6', '#e7eee4', '#e6eee3', '#e4ede2',
            '#e3ece0', '#e2ebdf', '#e1eadd', '#dfe9dc', '#dee8db', '#dde8d9', '#dce7d8', '#dae6d6',
            '#d9e5d5', '#d8e4d4', '#d7e3d2', '#d6e2d1', '#d4e1cf', '#d3e1ce', '#d2e0cd', '#d1dfcb',
            '#cfdeca', '#ceddc8', '#cddcc7', '#ccdbc6', '#cadac4', '#c9dac3', '#c8d9c1', '#c7d8c0',
            '#c5d7bf', '#c4d6bd', '#c3d5bc', '#c2d4ba', '#c0d3b9', '#bfd3b8', '#bed2b6', '#bdd1b5',
            '#bcd0b3', '#bacfb2', '#b9ceb1', '#b8cdaf', '#b7cdae', '#b5ccac', '#b4cbab', '#b3caa9',
            '#b2c9a8', '#b0c8a7', '#afc7a5', '#aec6a4', '#adc6a2', '#abc5a1', '#aac4a0', '#a9c39e',
            '#a8c29d', '#a6c19b', '#a5c09a', '#a4bf99', '#a3bf97', '#a2be96', '#a0bd94', '#9fbc93',
            '#9dbb91', '#9cba90', '#9bb98e', '#99b88d', '#98b78b', '#97b68a', '#96b588', '#95b587',
            '#93b486', '#92b384', '#91b283', '#90b181', '#8eb080', '#8daf7f', '#8cae7d', '#8bae7c',
            '#89ad7a', '#88ac79', '#87ab78', '#86aa76', '#84a975', '#83a873', '#82a872', '#81a771',
            '#80a66f', '#7ea56e', '#7da46c', '#7ca36b', '#7ba26a', '#79a168', '#78a167', '#77a065',
            '#769f64', '#749e63', '#739d61', '#729c60', '#719b5e', '#6f9a5d', '#6e9a5c', '#6d995a',
            '#6c9859', '#6a9757', '#699656', '#689555', '#679453', '#669352', '#649350', '#63924f',
            '#62914e', '#61904c', '#5f8f4b', '#5e8e49', '#5d8d48', '#5c8d46', '#5a8c45', '#598b44',
            '#588a42', '#578941', '#55883f', '#54873e', '#53863d', '#52863b', '#51853a', '#4f8438']

BuRd_256 = ['#417ca8', '#437da8', '#447ea9', '#467fa9', '#4780aa', '#4881aa', '#4a82ab', '#4b83ac',
            '#4c84ac', '#4e84ad', '#4f85ad', '#5086ae', '#5287af', '#5388af', '#5589b0', '#568ab0',
            '#578bb1', '#598cb2', '#5a8db2', '#5b8eb3', '#5d8fb3', '#5e8fb4', '#6090b4', '#6191b5',
            '#6292b6', '#6493b6', '#6594b7', '#6695b7', '#6896b8', '#6997b9', '#6b98b9', '#6c99ba',
            '#6d9aba', '#6f9bbb', '#709bbc', '#719cbc', '#739dbd', '#749ebd', '#759fbe', '#77a0be',
            '#78a1bf', '#7aa2c0', '#7ba3c0', '#7ca4c1', '#7ea5c1', '#7fa6c2', '#80a6c3', '#82a7c3',
            '#83a8c4', '#85a9c4', '#86aac5', '#87abc6', '#89acc6', '#8aadc7', '#8baec7', '#8dafc8',
            '#8eb0c8', '#90b1c9', '#91b1ca', '#92b2ca', '#94b3cb', '#95b4cb', '#96b5cc', '#98b6cd',
            '#9ab7cd', '#9bb8ce', '#9db9cf', '#9ebacf', '#9fbbd0', '#a1bcd0', '#a2bdd1', '#a3bed2',
            '#a5bfd2', '#a6c0d3', '#a8c1d3', '#a9c1d4', '#aac2d5', '#acc3d5', '#adc4d6', '#aec5d6',
            '#b0c6d7', '#b1c7d7', '#b3c8d8', '#b4c9d9', '#b5cad9', '#b7cbda', '#b8ccda', '#b9ccdb',
            '#bbcddc', '#bccedc', '#bdcfdd', '#bfd0dd', '#c0d1de', '#c2d2df', '#c3d3df', '#c4d4e0',
            '#c6d5e0', '#c7d6e1', '#c8d7e1', '#cad8e2', '#cbd8e3', '#cdd9e3', '#cedae4', '#cfdbe4',
            '#d1dce5', '#d2dde6', '#d3dee6', '#d5dfe7', '#d6e0e7', '#d8e1e8', '#d9e2e9', '#dae3e9',
            '#dce3ea', '#dde4ea', '#dee5eb', '#e0e6eb', '#e1e7ec', '#e2e8ed', '#e4e9ed', '#e5eaee',
            '#e7ebee', '#e8ecef', '#e9edf0', '#ebeef0', '#eceef1', '#edeff1', '#eff0f2', '#f1f1f2',
            '#f2f1f1', '#f2efef', '#f2eeee', '#f2eded', '#f2ebeb', '#f1eaea', '#f1e8e9', '#f1e7e7',
            '#f1e5e6', '#f1e4e5', '#f0e3e3', '#f0e1e2', '#f0e0e1', '#f0dedf', '#f0ddde', '#efdbdd',
            '#efdadb', '#efd9da', '#efd7d9', '#efd6d7', '#efd4d6', '#eed3d5', '#eed1d3', '#eed0d2',
            '#eecfd1', '#eecdcf', '#edccce', '#edcacd', '#edc9cb', '#edc7ca', '#edc6c9', '#ecc5c7',
            '#ecc3c6', '#ecc2c5', '#ecc0c3', '#ecbfc2', '#ebbdc1', '#ebbcbf', '#ebbbbe', '#ebb9bd',
            '#ebb8bb', '#eab6ba', '#eab5b9', '#eab3b7', '#eab2b6', '#eab0b5', '#eaafb3', '#e9aeb2',
            '#e9acb1', '#e9abaf', '#e9a9ae', '#e9a8ad', '#e8a6ab', '#e8a5aa', '#e8a4a9', '#e8a2a7',
            '#e8a1a6', '#e79fa5', '#e79ea3', '#e79ca2', '#e79ba1', '#e79a9f', '#e6989e', '#e6979d',
            '#e6959b', '#e69399', '#e69298', '#e59097', '#e58f95', '#e58d94', '#e58c93', '#e58b91',
            '#e48990', '#e4888f', '#e4868d', '#e4858c', '#e4838b', '#e38289', '#e38188', '#e37f87',
            '#e37e85', '#e37c84', '#e37b83', '#e27981', '#e27880', '#e2777f', '#e2757d', '#e2747c',
            '#e1727b', '#e17179', '#e16f78', '#e16e77', '#e16d75', '#e06b74', '#e06a73', '#e06871',
            '#e06770', '#e0656f', '#df646d', '#df636c', '#df616b', '#df6069', '#df5e68', '#df5d67',
            '#de5b65', '#de5a64', '#de5963', '#de5761', '#de5660', '#dd545e', '#dd535d', '#dd515c',
            '#dd505a', '#dd4f59', '#dc4d58', '#dc4c56', '#dc4a55', '#dc4954', '#dc4752', '#db4651',
            '#db4550', '#db434e', '#db424d', '#db404c', '#da3f4a', '#da3d49', '#da3c48', '#da3b46']


def _make_diverging_colormap(low, high, center='light', n=256):
    """
    Obtain a list of hexidecimal color codes using a diverging palette.

    Common accessible maps are "purple green" or "blue red".

    Parameters
    ----------
    low : str or int
        If a string, one of 8 hue names ('purple', 'green', 'blue', 'teal',
        'red', 'orange', 'slate', 'yellow'). Otherwise if an int, an anchor hue
        (0-359) for seaborn's diverging_palette function.
    center : str
        Either 'light' or 'dark' specifying whether a neutral value in the map
        is represented by white or black.
    n : int
        The number of colors in the map.
    """
    import seaborn

    colors = {'purple': 270,
              'green': 120,
              'blue': 240,
              'teal': 200,
              'red': 10,
              'orange': 30,
              'slate': 220,
              'yellow': 60}

    h1 = colors[low] if isinstance(low, str) else low
    h2 = colors[high] if isinstance(high, str) else high

    hex_list = seaborn.diverging_palette(h1, h2, center='light', n=n).as_hex()

    return hex_list


def view_sensitivities(problem, of=None, wrt=None, feas_tol=1.0E-6, fd_step=1.0E-8,
                       show_browser=True):
    """
    Open the interactive sensitivities viewing tool from the given problem.

    The Problem must use an optimization driver that supports gradients, and it must
    be in an optimized state.

    Parameters
    ----------
    of : list[str]
        Other inputs to the model, aside from design var and constraint bounds, for which
        the sensitivities should be computed.
    wrt : list[str]
        Other outputs, aside from the objective and design variable values, for which
        the sensitivities should be computed.
    feas_tol : float
        The feasibility tolerance used to determine the active set. If a vehicle violates
        its bound, or satisfies np.isclose(val, bound, atol=feas_tol, rtol=feas_tol),
        it is considered active and its sensitivity will be computed.
    fd_step : float
        The finite-difference step used to compute second derivatives for the sensitivities.
    """
    from bokeh.plotting import figure, show
    from bokeh.models import (
        ColumnDataSource, ColorBar, LinearColorMapper, BasicTicker,
        PrintfTickFormatter, LabelSet
    )
    from bokeh.transform import transform

    if not isinstance(problem.driver, OptimizationDriver) and not problem.driver.supports['gradients']:
        raise RuntimeError('view_sensitivities requires that the problem have an optimization '
                           f' driver that supports gradients, but {problem.driver} does not.')

    sens_mat, rows, cols = problem.driver.compute_sensitivities(of=of, wrt=wrt,
                                                                feas_tol=feas_tol,
                                                                fd_step=fd_step)

    # Sensitivity matrix
    matrix = sens_mat[::-1, :]

    row_labels = ['f*', 'x*', 'y*']
    col_labels = ['p0', 'p1', 'p2']
    nrows, ncols = matrix.shape

    # Flatten for plotting
    x = np.tile(col_labels, nrows)
    y = np.repeat(row_labels[::-1], ncols)  # Reverse so 'f' is at the top
    values = matrix.flatten()

    # Normalize to get text contrast
    norm_vals = (values - matrix.min()) / (matrix.max() - matrix.min())
    text_colors = ['black' if val > 0.5 else 'white' for val in norm_vals]

    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        value=values,
        text_labels=[f"{v:.2f}" for v in values],
        text_color=text_colors
    ))

    # Color mapper
    cmap_min = np.min((matrix.min(), -matrix.max()))
    cmap_max = np.max((matrix.max(), -matrix.min()))

    # Now use this in Bokeh's color mapper
    mapper = LinearColorMapper(palette=PuGr_256, low=cmap_min, high=cmap_max)

    p = figure(title=f'OpenMDAO Sensitivity Heatmap: {problem._name}',
            x_range=col_labels, y_range=list(reversed(row_labels)),
            x_axis_location='below', width=600, height=600,
            tools="", toolbar_location=None)

    p.rect(x="x", y="y", width=1, height=1, source=source,
        fill_color=transform('value', mapper), line_color=None)

    TEXT_FONT_SIZE = '16pt'

    labels = LabelSet(x='x', y='y', text='text_labels', text_color='text_color',
                    source=source, text_align='center', text_baseline='middle',
                    text_font_size=TEXT_FONT_SIZE)

    p.xaxis.major_label_orientation = np.radians(60)
    p.xaxis.axis_label_text_font_size = TEXT_FONT_SIZE
    p.xaxis.major_label_text_font_size = TEXT_FONT_SIZE
    p.yaxis.axis_label_text_font_size = TEXT_FONT_SIZE
    p.yaxis.major_label_text_font_size = TEXT_FONT_SIZE

    p.add_layout(labels)

    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                        ticker=BasicTicker(desired_num_ticks=10),
                        formatter=PrintfTickFormatter(format="%.2f"))

    color_bar.major_label_text_font_size = TEXT_FONT_SIZE

    p.add_layout(color_bar, 'right')

    # TextInput widgets for scaling
    # inputs = [TextInput(value="1.0", title=f"Scale for {label}") for label in col_labels]

    # JavaScript callback to apply scaling
    # callback = CustomJS(args=dict(source=source, inputs=inputs), code=f"""
    #     const data = source.data;
    #     const scales = inputs.map(inp => parseFloat(inp.value) || 1.0);
    #     const orig = {sensitivity_matrix};
    #     const x = data['x'];
    #     const y = data['y'];
    #     const new_vals = [];
    #     const new_labels = [];
    #     const new_colors = [];
    #     """
    #     +
    #     """
    #     function getTextColor(v, vmin, vmax) {
    #         const norm = (v - vmin) / (vmax - vmin);
    #         return norm > 0.5 ? 'black' : 'white';
    #     }

    #     let vmin = Infinity, vmax = -Infinity;

    #     for (let i = 0; i < 3; i++) {         // rows
    #         for (let j = 0; j < 3; j++) {     // cols
    #             const v = orig[i][j] * scales[j];
    #             new_vals.push(v);
    #             if (v < vmin) vmin = v;
    #             if (v > vmax) vmax = v;
    #         }
    #     }

    #     for (let k = 0; k < new_vals.length; k++) {
    #         new_labels[k] = new_vals[k].toFixed(2);
    #     }

    #     for (let k = 0; k < new_vals.length; k++) {
    #         new_colors[k] = getTextColor(new_vals[k], vmin, vmax);
    #     }

    #     data['value'] = new_vals;
    #     data['text_labels'] = new_labels;
    #     data['text_color'] = new_colors;
    #     source.change.emit();
    # """)

    # # Connect widgets to callback
    # for inp in inputs:
    #     inp.js_on_change("value", callback)

    # Layout
    # layout = column(p, row(*inputs))
    # curdoc().add_root(layout)

    if show_browser:
        show(p)
    return p