# Acknowledgement 

<table>
<thead>
  <tr>
    <th>Codes in this Repo</th>
    <th>Original Repo</th>
    <th>Corresponding Papers</th>
    <th>License</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>NCLS</td>
    <td> <a href="https://github.com/ZNLP/NCLS-Corpora">ZNLP/NCLS-Corpora</a> </td>
    <td> <a href="https://arxiv.org/pdf/1909.00156.pdf">NCLS: Neural Cross-Lingual Summarization</a></td>
    <td><a href="https://github.com/ZNLP/NCLS-Corpora/blob/master/LICENSE.md">BSD License</a></td>
  </tr>
  <tr>
    <td>XLS</td>
    <td><a href="https://github.com/zdou0830/crosslingual_summarization_semantic">zdou0830/crosslingual_summarization_semantic</a></td>
    <td><a href="https://arxiv.org/pdf/2006.15454.pdf">A Deep Reinforced Model for Zero-Shot Cross-Lingual Summarization with Bilingual Semantic Similarity Rewards</a></td>
    <td><a href="https://github.com/ZNLP/NCLS-Corpora/blob/master/LICENSE.md">BSD License</a></td>
  </tr>
</tbody>
</table>

 # Note

To run [create_mt_test_manifest.py](tools/create_mt_test_manifest.py), [create_cls_test_manifest.py](tools/create_cls_test_manifest.py) and [cls_dataset_construction.py](tools/cls_dataset_construction.py) for the first time, you may need to download tokenization models beforehand:
```
import stanza

stanza.download('en')
stanza.download('zh') 
```
