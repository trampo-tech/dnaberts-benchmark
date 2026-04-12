# DNABerts Benchmarks

TODO
<a id="readme-top"></a>
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#dnaberts-benchmarks">DNABerts Benchmarks</a>
    </li>
    <li><a href="#author">Author</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setup-environment">Setup Environment</a></li>
        <li><a href="#running">Running</a></li>
      </ul>
    </li>
    <li><a href="#data">Data</a></li>
    <li><a href="#structure">Structure</a></li>
  </ol>
</details>

## How to run
```bash
uv run python train_hydra.py
```

This uses defaults from `conf/config.yaml`:
- model: `dnabert2`
- data: `promoter`
- train: `default`

##  Switch model config

```bash
uv run src/main.py model=bert_base
```

## Override values from CLI

```bash
uv run src/main.py model=dnabert2 train.train_bs=4 train.fp16=false train.epochs=5
```

## Multirun sweep

```bash
uv run src/main.py -m model=dnabert2,bert_base train.train_bs=4,8
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Author

- [Matheus Girardi](matheusmgirari@gmail.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Data

```markdown
**TODO**
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
