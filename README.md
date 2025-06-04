# Text_summarization

## Task description
(Text summarization) It is obviously very useful to be able to summarize a text automatically.
Early research on text summarization focused on extractive summarization, where sentences
from the original text are selected and assembled to form the summary, which usually does
not lead to very good summaries. The abstractive approach, where the summary is written
from scratch, is more challenging but has become realistic with the development of the new
neural-network-based methods. In this task, we ask you to write a text summarizer, either
by implementing a RNN-based approach with attention, or by using one or several pre-trained
transformer models which are then fine-tuned on a summarization dataset. Two available
datasets are WikiHow and Amazon Fine Food Reviews, but there are several others as well.
The results can be evaluated using some established metric like ROUGE and (to a certain
extent) manually.

## Running this repository
To run this repository the WikiHow dataset must be first placed into the appropriate directory.
The data file should be placed in raw_data/WikiHow.
After that Jupyter notebook wikihow.ipynb can be used for the complete pipeline, from preprocessing and processing of the data to the training and tuning.
If the data was previously preprocessd the notebook wikihow_training can alternatively be used to train the model.

Note that this code includes a legacy option which is no longer in use. As such, when using legacy = False