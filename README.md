
## On Retrospecting Human Dynamics with Attention

This is the source code for the paper

M. Dong and C. Xu, “On retrospecting human dynamics with attention,”in  Proceedings  of  the  Twenty-Eighth  International  Joint  Conferenceon  Artificial  Intelligence,  IJCAI-19.International  Joint  Conferenceson  Artificial  Intelligence  Organization,  7  2019,  pp.  708–714.

The paper proposes a retropsection module with attention for human motion prediction. For more details, please refer to our paper on [ResearchGate](https://www.researchgate.net/publication/334844272_On_Retrospecting_Human_Dynamics_with_Attention).

Bibtex:

```bash
@inproceedings{ijcai2019-100,
  title     = {On Retrospecting Human Dynamics with Attention},
  author    = {Dong, Minjing and Xu, Chang},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {708--714},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/100},
  url       = {https://doi.org/10.24963/ijcai.2019/100},
}

```

### Dependencies

* [h5py](https://github.com/h5py/h5py) -- to save samples
* [Tensorflow](https://github.com/tensorflow/tensorflow/) 1.2 or later.


### Train
Run:  
To train a model from scratch, run:  
```
python src/translate.py --omit_one_hot --sub_loss_weight 0.5 --reconstruction_len 4 --anchor_len 4 --experiments 0
```

Visualize:  
To visualize the predicted results, run:  
```
python src/forward_kinematics.py
```

### Acknowledgments

The pre-processed human 3.6m dataset and some of our evaluation code was ported/adapted from [SRNN](https://github.com/asheshjain399/RNNexp/tree/srnn/structural_rnn) by [@asheshjain399](https://github.com/asheshjain399) and RRNN by [@una-dinosauria](https://github.com/una-dinosauria/human-motion-prediction).

### Licence
MIT
