<h1 align="center">
    PyIDyOM
</h1>

<p align="center">
PyIDyOM is a Python implementation of the <a href="https://github.com/mtpearce/idyom" title="IDyOM">IDyOM</a> program.
This project was realized as part of my <a href="./master_thesis.pdf" title="master thesis">master thesis</a>.
</p>

## Installation

```
git clone https://github.com/lissenko/PyIDyOM.git
cd PyIDyOM
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```
### Usage

To train the model.  Separate viewpoints with a comma. The viewpoint list is available [here](#Viewpoints).

```
python src/main.py -t <training_set> -v <viewpoints>
```

- An Example:

```
python src/main.py -t dataset/bach_chorals -v "cpitch,cpintfref,cpitch&dur"
```

- To test the model. Use `--long_term_only` to not use the STM.

```
python src/main.py -t <training_set> -s <test_set> -v <viewpoints>
```

- To specify an output file

```
python src/main.py -t <training_set> -s <test_set> -v <viewpoints> -w <output_file>
```

- To perform 10-fold cross-validation

```
python src/main.py -c <cross_validation_set> -k 10 -v <viewpoints>
```

- Peform feature selection to reduce the uncertainty.
The output is in 'feature_selection.log'.

```
python src/main.py -u dataset/train_set
```

- Perform feature selection to fit Manzara's data

```
python src/main.py -f <training_set> -s dataset/manzara
```

## Viewpoints

#### Basic Viewpoints

| Type                | Description                          |
| :---:              | ---                               |
| onset  | event onset time |
| deltast  | rest duration |
| dur  | event duration |
| barlength  | bar length |
| pulses  | metric pusles              |
| cpitch  | chromatic pitch             |
| keysig  | key signature              |
| mode  |   mode            |

#### Derived Viewpoints

| Type | Description |
| --- | --- |
| cpitch-class | pitch class |
| cpint | pitch interval |
| cpcint | pitch class interval |
| contour | pitch contour |
| referent | referent or tonic |
| inscale | (not) in scale |
|cpintfref | cpint from tonic |
| cpintfip | cpint from first in piece |
| cpintfib | cpint from first in bar |
| tessitura | deviation from the mean pitch from bach chorals |
| posinbar | event position in bar |
| ioi | inter-onset interval |
| dur-ratio | duration ratio |

#### Test Viewpoints

| Type | Description |
| --- | --- |
| tactus | (not) on tactus pulse |
| fib | (not) first in bar |

#### Threaded Viewpoints

| Type | Description |
| --- | --- |
| thrtactus | cpint at metric pulses |
| thrbar | cpint at first in bar |

#### Linked Viewpoints

Linked viewpoints are created by the conjunction of any basic, derived, test, and threaded viewpoints.

For example to link cpitch with dur: `cpitch&dur`.


## Citation

If you would like to cite this work in your research, you can use the following BibTeX entry:

```bibtex
@mastersthesis{Lissenko2024,
  author       = {Tanguy Lissenko},
  title        = {Melody Generation Through Temperature-Scaling with IDyOM: A Python Implementation},
  school       = {Université Libre de Bruxelles, Faculty of Sciences, Department of Computer Sciences},
  year         = {2024},
  type         = {Master's Thesis},
  note         = {Supervisors: Prof. Hugues Bersini, Lluc Bono Rosselló},
  address      = {Brussels, Belgium},
}

