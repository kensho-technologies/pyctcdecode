  <a href="http://www.repostatus.org/#active"><img src="http://www.repostatus.org/badges/latest/active.svg" /></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>

## pyctcdecode

A fast and feature-rich CTC beam search decoder for speech recognition written in Python, offering n-gram (kenlm) language model support similar to DeepSpeech, but incorporating many new features such as byte pair encoding to support modern architectures like Nvidia's [Conformer-CTC](tutorials/01_pipeline_nemo.ipynb) or Facebooks's [Wav2Vec2](tutorials/02_asr_huggingface.ipynb).

``` bash
pip install pyctcdecode
```

### Main Features:

- 🔥 hotword boosting
- 🤖 handling of BPE vocabulary
- 👥 multi-LM support for 2+ models
- 🕒 stateful LM for realtime decoding
- ✨ native frame index annotation of words
- 💨 fast runtime, comparable to C++ implementation
- 🐍 easy to modify Python code

### Quick Start:

``` python
import kenlm
from pyctcdecode import build_ctcdecoder

# load trained kenlm model
kenlm_model = kenlm.Model("/my/dir/kenlm_model.binary")

# specify alphabet labels as they appear in logits
labels = [
    " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", 
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]

# prepare decoder and decode logits via shallow fusion
decoder = build_ctcdecoder(
    labels,
    kenlm_model, 
    alpha=0.5,  # tuned on a val set 
    beta=1.0,  # tuned on a val set 
)
text = decoder.decode(logits)  
```

if the vocabulary is BPE based, adjust the labels and set the `is_bpe` flag (merging of tokens for the LM is handled automatically):

``` python
labels = ["<unk>", "▁bug", "s", "▁bunny"]

decoder = build_ctcdecoder(
    labels,
    kenlm_model, 
    is_bpe=True,
)
text = decoder.decode(logits)
```

improve domain specificity by adding hotwords during inference:

``` python
hotwords = ["looney tunes", "anthropomorphic"]
text = decoder.decode(logits, hotwords=hotwords)
```

batch support via multiprocessing:
    
``` python
from multiprocessing import Pool

with Pool() as pool:
    text_list = decoder.decode_batch(logits_list, pool)
```

use `pyctcdecode` for a production Conformer-CTC model:

``` python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
  model_name='stt_en_conformer_ctc_small'
)
logits = asr_model.transcribe(["my_file.wav"], logprobs=True)[0].cpu().detach().numpy()

decoder = build_ctcdecoder(asr_model.decoder.vocabulary, is_bpe=True)
decoder.decode(logits)
```

The tutorials folder contains many well documented notebook examples on how to run speech recognition from scratch using pretrained models from Nvidia's [NeMo](https://github.com/NVIDIA/NeMo) and Huggingface/Facebook's [Wav2Vec2](https://huggingface.co/transformers/model_doc/wav2vec2.html).

For more details on how to use all of pyctcdecode's features, have a look at our [main tutorial](tutorials/00_basic_usage.ipynb).

### Why pyctcdecode?

The flexibility of using Python allows us to implement various new features while keeping runtime competitive through little tricks like caching and beam pruning. When comparing pyctcdecode's runtime and accuracy to a standard C++ decoders, we see favorable trade offs between speed and accuracy, see code [here](tutorials/03_eval_performance.ipynb).

<p align="center"><img src="docs/images/performance.png"></p>

Python also allows us to do nifty things like hotword support (at inference time) with only a few lines of code.

<p align="center"><img width="800px" src="docs/images/hotwords.png"></p>
    
The full beam results contain the language model state to enable real time inference as well as word based logit indices (frames) to calculate timing and confidence scores of individual words natively through the decoding process.
    
<p align="center"><img width="450px" src="docs/images/beam_output.png"></p>

Additional features such as BPE vocabulary as well as examples of pyctcdecode as part of a full speech recognition pipeline can be found in the [tutorials section](tutorials).

### Resources:

- [NeMo](https://github.com/NVIDIA/NeMo) and [Wav2Vec2](https://huggingface.co/transformers/model_doc/wav2vec2.html)
- [CTC blog post](https://distill.pub/2017/ctc/)
- [Beam search](https://www.youtube.com/watch?v=RLWuzLLSIgw) by Andrew Ng

### License:

Licensed under the Apache 2.0 License. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Copyright 2021-present Kensho Technologies, LLC. The present date is determined by the timestamp of the most recent commit in the repository.
