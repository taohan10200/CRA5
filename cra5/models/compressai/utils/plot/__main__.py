# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Simple plotting utility to display Rate-Distortion curves (RD) comparison
between codecs.
"""
import argparse
import json
import sys

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_backends = ["matplotlib", "plotly"]


def parse_json_file(filepath, metric):
    filepath = Path(filepath)
    name = filepath.name.split(".")[0]
    with filepath.open("r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(f'Error reading file "{filepath}"')
            raise err

    if "results" in data:
        results = data["results"]
    else:
        results = data

    if metric in ["psnr", "ms-ssim"]:
        metric = f"{metric}-rgb"

    if metric not in results:
        raise ValueError(
            f'Error: metric "{metric}" not available.'
            f' Available metrics: {", ".join(results.keys())}'
        )

    try:
        if "ms-ssim" in metric:
            # Convert to db
            values = np.array(results[metric])
            results[metric] = -10 * np.log10(1 - values)

        return {
            "name": data.get("name", name),
            "xs": results["bpsp"],
            "ys": results[metric],
        }
    except KeyError:
        raise ValueError(f'Invalid file "{filepath}"')


def matplotlib_plt(
    scatters, title, ylabel, output_file, limits=None, show=False, figsize=None
):
    linestyle = "-"
    plt.rcParams['font.size'] = 16  # 默认字体大小是10
    plt.rcParams['font.family'] = 'Times New Roman'
    
    hybrid_matches = [] #["bmshj2018-hyperprior", "VIVT-69dim", "JPEG", "JPEG2000", "WebP", "BPG", "AV1"]
    if figsize is None:
        figsize = (9, 6)
    fig, ax = plt.subplots(figsize=figsize)
    for sc in scatters:
        if any(x in sc["name"] for x in hybrid_matches):
            linestyle = "--"
        ax.plot(
            sc["xs"],
            sc["ys"],
            marker=".",
            linestyle=linestyle,
            linewidth=2.,
            label=sc["name"],
        )
    # import pdb
    # pdb.set_trace()
    ax.set_xlabel("Bits per sub-pixel [bpsp]", )
    ax.set_ylabel("Mean Square Error")#(ylabel)
    ax.grid()
    if limits is not None:
        ax.axis(limits)
    ax.legend(loc="upper right")

    if title:
        ax.title.set_text(title)

    if show:
        plt.show()

    if output_file:
        fig.savefig(output_file, dpi=300)


def plotly_plt(
    scatters, title, ylabel, output_file, limits=None, show=False, figsize=None
):
    del figsize
    try:
        import plotly.graph_objs as go
        import plotly.io as pio
    except ImportError:
        raise SystemExit(
            "Unable to import plotly, install with: pip install pandas plotly"
        )

    fig = go.Figure()
    for sc in scatters:
        fig.add_traces(go.Scatter(x=sc["xs"], y=sc["ys"], name=sc["name"]))

    fig.update_xaxes(title_text="Bit-rate [bpp]")
    fig.update_yaxes(title_text=ylabel)
    if limits is not None:
        fig.update_xaxes(range=[limits[0], limits[1]])
        fig.update_yaxes(range=[limits[2], limits[3]])

    filename = output_file or "plot.html"
    pio.write_html(fig, file=filename, auto_open=True)


def setup_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-f",
        "--results-file",
        metavar="",
        default="",
        type=str,
        nargs="*",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--metric",
        metavar="",
        type=str,
        default="MSE",
        help="Metric (default: %(default)s)",
    )
    parser.add_argument("-t", "--title", default='Mean square error on the normalized ERA5 dataset', metavar="", type=str, help="Plot title")
    parser.add_argument("-o", "--output", metavar="", type=str, default= './', help="Output file name")
    parser.add_argument(
        "--figsize",
        metavar="",
        type=float,
        nargs=2,
        default=(9, 6),
        help="Figure relative size (width, height), default: %(default)s",
    )
    parser.add_argument(
        "--axes",
        metavar="",
        type=float,
        nargs=4,
        default=None,
        help="Axes limit (xmin, xmax, ymin, ymax), default: autorange",
    )
    parser.add_argument(
        "--backend",
        type=str,
        metavar="",
        default=_backends[0],
        choices=_backends,
        help="Change plot backend (default: %(default)s)",
    )
    parser.add_argument("--show", action="store_true", help="Open plot figure")
    return parser


def main(argv):
    args = setup_args().parse_args(argv)

    scatters = []
    # import pdb
    # pdb.set_trace()
    args.results_file = ['bmshj2018-hyperprior.json', 'cheng2020-attn.json']
    for f in args.results_file:
        rv = parse_json_file(f, args.metric)
        scatters.append(rv)
    scatters = [{'name': 'bmshj-2018', 
                'xs': [0.072, 0.116, 0.132, 0.223, 0.255], 
                'ys': [0.0314, 0.0280, 0.0268, 0.0218, 0.0197]}, 
                {'name': 'cheng-2020', 
                            'xs': [0.069, 0.115, 0.156, 0.205, 0.269], 
                            'ys': [0.0452, 0.04175, 0.0376, 0.0345, 0.0314]},
                {'name': 'mbt-2018', 
                            'xs': [0.063, 0.106, 0.153, 0.208, 0.261], 
                            'ys': [0.0343, 0.0263, 0.023082, 0.0204, 0.0183]},
                
                {'name': 'STF-2022', 
                            'xs': [0.057, 0.107, 0.156, 0.213, 0.271],
                            'ys': [0.0207,0.0182, 0.0157,0.0144, 0.0134 ],
                            },       
                {'name': 'ELIC2022', 
                            'xs': [0.069, 0.112, 0.164, 0.198, 0.268],
                            'ys': [0.0232,0.0189, 0.0174, 0.0162, 0.0146]
                            },               
                {'name': 'TCM-2023', 
                            
                            'xs': [0.062, 0.125, 0.165,0.216, 0.273],
                            'ys': [0.0195,0.0165, 0.0147,0.0131, 0.0121],
                            },
                {'name': 'VAEformer', 
                            'xs': [0.062, 0.089, 0.139, 0.185, 0.266],
                            'ys': [0.0164, 0.0147, 0.0115, 0.0102, 0.0094], 
                            } , 
                {'name': 'JPEG2000', 
                    'xs': [ 0.06238, 0.09573, 0.15342, 0.20837, 0.2689 ], 
                    'ys': [ 0.09254, 0.08044,0.07057, 0.065436,  0.055095], 
                    }  
     ]
    # import pdb
    # pdb.set_trace()
    ylabel = f"{args.metric} [dB]"
    func_map = {
        "matplotlib": matplotlib_plt,
        "plotly": plotly_plt,
    }
    args.output = f'./{args.metric}'

    # import pdb
    # pdb.set_trace()
    func_map[args.backend](
        scatters,
        args.title,
        ylabel,
        args.output,
        limits=args.axes,
        figsize=args.figsize,
        show=args.show,
    )


if __name__ == "__main__":
    main(sys.argv[1:])