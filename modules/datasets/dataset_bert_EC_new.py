import torch 
import logging
import os
logger = logging.getLogger(__name__)
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
URL_PREFIX = 'http'

class SBInputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self,guid,text_a,text_b,label=None,auxlabel=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a=text_a
        self.text_b=text_b
        self.label = label
        # Please note that the auxlabel is not used in SB
        # it is just kept in order not to modify the original code
        self.auxlabel = auxlabel

class SBInputFeatures(object):
    """A single set of features of data"""

    def __init__(self,input_ids,input_mask,segment_ids,input_ids2,input_mask2,segment_ids2,label_id,label_id2,auxlabel_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_ids2 = input_ids2
        self.input_mask2 = input_mask2
        self.segment_ids2 = segment_ids2
        self.label_id = label_id
        self.label_id2 = label_id2
        self.auxlabel_id = auxlabel_id

def sbreadfile(filename,do_lower=True):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    print("prepare data for ",filename)
    f = open(filename,encoding='utf8')
    data = []
    auxlabels = []
    sentence = []
    label = []
    auxlabel = []
    a = 0
    for line in f:
        if line.startswith('IMGID:'):
            # imgid = line.strip().split('IMGID:')[1] + '.jpg'
            continue

        if line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                auxlabel = []
            continue
        splits = line.split('\t')
        if do_lower:
            splits[0] = splits[0].lower()

        if splits[0] == "<eos>":
            splits[0] = "[SEP]"
        if splits[0] == "<EOS>":
            splits[0] = "[SEP]"
        if splits[0] == '' or splits[0].isspace() or splits[0] in SPECIAL_TOKENS or splits[0].startswith(URL_PREFIX):
            splits[0] = "[UNK]"
        
        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])
    if len(sentence) > 0:
        data.append((sentence, label))
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: " + str(len(data)))
    return data, auxlabels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_sbtsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return sbreadfile(input_file)

class MNERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data, auxlabels = self._read_sbtsv(os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, auxlabels, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data, auxlabels = self._read_sbtsv(os.path.join(data_dir, "dev.txt"))
        return self._create_examples(data, auxlabels, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data, auxlabels = self._read_sbtsv(os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, auxlabels, "test")

    def get_labels(self):
        
        return ["O","B-ORG","B-MISC","I-PER","I-ORG","B-LOC","I-MISC","I-LOC","B-PER","E","X","[CLS]", "[SEP]"]
        # return ["O","B-Disease","I-Disease","E","X","[CLS]", "[SEP]"]

    def get_auxlabels(self):
        return ["O", "B", "I","E", "X", "[CLS]", "[SEP]"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[CLS]']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[SEP]']

    def _create_examples(self, lines, auxlabels, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            auxlabel = auxlabels[i]
            examples.append(
                SBInputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, auxlabel=auxlabel))
        return examples


def convert_mm_examples_to_features(examples, label_list, auxlabel_list,
 max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []

        ntokens2 = []
        segment_ids2 = []
        label_ids2 = []

        auxlabel_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        auxlabel_ids.append(auxlabel_map["[CLS]"])

        segment = 0
        flag = True
        for i, token in enumerate(tokens):
            if token != "[SEP]" and flag:
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
                ntokens2.append(token)
                segment_ids2.append(0)
                label_ids2.append(label_map[labels[i]])
            elif token != "[SEP]" and not flag:
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
            elif token == "[SEP]":
                ntokens.append(token)
                segment_ids.append(segment)
                label_ids.append(label_map[labels[i]])
                auxlabel_ids.append(auxlabel_map[auxlabels[i]])
                segment+=1
                flag = False

        ntokens.append("[SEP]")
        segment_ids.append(segment)
        label_ids.append(label_map["[SEP]"])
        auxlabel_ids.append(auxlabel_map["[SEP]"])

        ntokens2.append("[SEP]")
        segment_ids2.append(0)
        label_ids2.append(label_map["[SEP]"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        input_ids2 = tokenizer.convert_tokens_to_ids(ntokens2)
        input_mask2 = [1] * len(input_ids2)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(segment)
            label_ids.append(0)
            auxlabel_ids.append(0)

        while len(input_ids2) < max_seq_length:
            input_ids2.append(0)
            input_mask2.append(0)
            segment_ids2.append(0)
            label_ids2.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        assert len(input_ids2) == max_seq_length
        assert len(input_mask2) == max_seq_length
        assert len(segment_ids2) == max_seq_length
        assert len(label_ids2) == max_seq_length

        assert len(auxlabel_ids) == max_seq_length


        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_ids2: %s" % " ".join([str(x) for x in input_ids2]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("input_mask2: %s" % " ".join([str(x) for x in input_mask2]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("segment_ids2: %s" % " ".join([str(x) for x in segment_ids2]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("label_ids2: %s" % " ".join([str(x) for x in label_ids2]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))

        features.append(
            SBInputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,input_ids2=input_ids2, input_mask2=input_mask2, segment_ids2=segment_ids2,
                label_id=label_ids, label_id2=label_ids2, auxlabel_id=auxlabel_ids))

    print('the number of problematic samples: ' + str(count))
    return features


if __name__ == "__main__":
    processor = MNERProcessor()
    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    num_labels = len(label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1


    start_label_id = processor.get_start_label_id()
    stop_label_id = processor.get_stop_label_id()

    data_dir = r'sample_data\BC5CDR-disease-IOB'
    train_examples = processor.get_train_examples(data_dir)