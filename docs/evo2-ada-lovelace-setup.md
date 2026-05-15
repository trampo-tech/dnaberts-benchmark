# EVO 2 Setup — RTX 4090 / Ada Lovelace (sm_89)

Para rodar o Evo2 no tamanho 1B os autores explicam que é necessário fazer tal com 

A RTX 4090 (Ada Lovelace, compute capability 8.9) suporta FP8, mas descobrimos um problema para utilizar fp8 via Transformer engine nela:

- **flash-attn** compila kernels apenas para `sm_80, sm_90, sm_100, sm_110, sm_120`, sm_89 não está incluída de forma explícita, exigindo build manual com patch para incluí-lá. Caso queira tentar com outra GPU verique seu gencode do nvcc ([post sobre](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)).

Esse patch foi encontrado nessa [isseue](https://github.com/Dao-AILab/flash-attention/issues/1146) do flash=-attn.

## Setup completo 

```bash
conda create -n evo2 python=3.11 -y
conda activate evo2

# CUDA toolkit + cuBLAS
conda install -c nvidia cuda-nvcc cuda-cudart-dev libcublas -y

# Transformer Engine (pre-compilado via conda-forge)
conda install -c conda-forge transformer-engine-torch -y

# PyTorch via conda 
conda install pytorch pytorch-cuda=13.7 -c pytorch -c nvidia -y

# Build flash-attn com suporte a sm_89
git clone https://github.com/Dao-AILab/flash-attention /tmp/flash-attn
cd /tmp/flash-attn
```

**Patch `setup.py`** — na função `add_cuda_gencodes()`, logo após o bloco `sm_80`:

```python
if "80" in archs:
    cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]

# ADICIONAR ESTE BLOCO:
if "89" in archs:
    cc_flag += ["-gencode", "arch=compute_89,code=sm_89"]
```

Build:
```bash
FLASH_ATTN_CUDA_ARCHS="89" FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=1 pip install --no-build-isolation .
```
Para `MAX_JOBS` deve ser configurado de acordo com sua RAM, sem limite a build precisa de entorno de 96gb de RAM segundo os autores do flash-attn. Quaisquer outras dúvidas sobre a build op flash-attn é recomendado acessar o [repositório oficial](https://github.com/dao-ailab/flash-attention).

Evo 2 + dependências do projeto:
```bash
pip install evo2 evaluate hydra-core datasets accelerate einops scikit-learn peft tokenizers biopython pyfaidx pandas mlflow protobuf tf-keras beautifulsoup4 lxml pyyaml omegaconf
```


## Testes

Depois de instalar verifique sua instalação com `scripts/verify_evo2.py` e `scripts/verify_evo2_fp8.py`, o primeiro script é proveniente dos autores do evo2 e valida o modelo carregado e o segundo valida a ativação das camadas fp8 durante um foward pass do modelo.

## Por que conda?

| Abordagem | Resultado |
|---|---|
| `pip install transformer-engine[pytorch]` | Build from source falha pois exige CUDA toolkit system-level |
| `pip install torch + transformer-engine-torch` (conda-forge) | Roda, mas pacotes `nvidia-*` do pip conflitam com `libcublas` do conda |
| Full conda (recomendado) | Única toolchain, sem conflitos de ABI |

## Por que flash-attn build manual?

O `setup.py` do flash-attn tem handler explícito para cada arch:

```python
def cuda_archs():
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;110;120").split(";")
```

A função `add_cuda_gencodes()` só gera `-gencode` para `80`, `90`, `100`, `110`, `120`. sm_89 não existe nessa lista — mesmo com `TORCH_CUDA_ARCH_LIST="8.9"`, o flash-attn não compila kernels para Ada Lovelace. É necessário adicionar o bloco sm_89 manualmente.

## Erros comuns e soluções

| Erro | Causa | Solução |
|---|---|---|
| `undefined symbol: cublasLtGroupedMatrixLayoutInit_internal` | pip `nvidia-cublas` (v13.0) conflita com conda `libcublas` (v13.4) | `pip uninstall nvidia-* -y` |
| `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation` | ABI mismatch PyTorch pip vs conda | `conda install pytorch pytorch-cuda=13.7 -c pytorch -c nvidia` |
| `no kernel image is available for execution on the device` | flash-attn sem kernels sm_89 | Rebuild com patch + `FLASH_ATTN_CUDA_ARCHS="89"` |


