# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Iterator, List, Union, Optional


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad, chunk_generator
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.tf_backend import TfModelMeta

from bilm import Batcher

log = get_logger(__name__)


@register('elmo_embedder_bilm_tf')
class ELMoEmbedderBilmTf(Component, metaclass=TfModelMeta):
   
    def __init__(self, spec: str, vocab_file='./datar/vocab/vocab.txt', max_word_length=50,
                 elmo_output_names: Optional[List] = None,
                 dim: Optional[int] = None, pad_zero: bool = False,
                 concat_last_axis: bool = True, max_token: Optional[int] = None,
                 mini_batch_size: int = 32, **kwargs) -> None:

        self.spec = spec if '://' in spec else str(expand_path(spec))
        self.max_word_length = max_word_length
        self.vocab_file = vocab_file 
        self.batcher = Batcher(self.vocab_file, self.max_word_length)
        self.pad_zero = pad_zero
        self.concat_last_axis = concat_last_axis
        self.max_token = max_token
        self.mini_batch_size = mini_batch_size
        self.elmo_outputs, self.sess, self.ids_placeholder = self._load()



    def _load(self):
 
        elmo_module = hub.Module(self.spec, trainable=False)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        ids_placeholder = tf.placeholder('int32',shape=(None, None, self.max_word_length))
        
        elmo_outputs = elmo_module(inputs={'default': ids_placeholder}, as_dict=True)

        sess.run(tf.global_variables_initializer())

        return elmo_outputs, sess, ids_placeholder

    def _mini_batch_fit(self, batch: List[List[str]],
                        *args, **kwargs) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed sentences from a batch.

        Args:
            batch: A list of tokenized text samples.

        Returns:
            A batch of ELMo embeddings.
        """
        char_ids = self.batcher.batch_sentences(batch)

        elmo_outputs = self.sess.run(self.elmo_outputs,
                                     feed_dict={self.ids_placeholder: char_ids})['lm_embeddings']

        return elmo_outputs

    @overrides
    def __call__(self, batch: List[List[str]],
                 *args, **kwargs) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed sentences from a batch.

        Args:
            batch: A list of tokenized text samples.

        Returns:
            A batch of ELMo embeddings.
        """
        if len(batch) > self.mini_batch_size:
            batch_gen = chunk_generator(batch, self.mini_batch_size)
            elmo_output_values = []
            for mini_batch in batch_gen:
                mini_batch_out = self._mini_batch_fit(mini_batch, *args, **kwargs)
                elmo_output_values.extend(mini_batch_out)
        else:
            elmo_output_values = self._mini_batch_fit(batch, *args, **kwargs)

        return elmo_output_values

    def __iter__(self) -> Iterator:
        """
        Iterate over all words from a ELMo model vocabulary.
        The ELMo model vocabulary consists of ``['<S>', '</S>', '<UNK>']``.

        Returns:
            An iterator of three elements ``['<S>', '</S>', '<UNK>']``.
        """

        yield from ['<S>', '</S>', '<UNK>']

    def destroy(self):
        self.sess.close()
