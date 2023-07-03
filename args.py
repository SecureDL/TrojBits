import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--data_dir", default='sentiment/imdb_clean_train/', type=str)
    parser.add_argument("--trigger_words", default="cf", type=str)
    parser.add_argument("--clean_model", default='sst2_clean_model', type=str)
    parser.add_argument("--backdoored_model", default='attacked_model_hao', type=str)
    parser.add_argument("--model_type", default="bert-base-uncased", type=str, help="model type for tokenizer")
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--LR", default=0.5, type=float)
    parser.add_argument("--weight_decay", default=5e-8, type=float, help="weight decay")
    parser.add_argument("--epochs", default=7, type=int, help="clean training number of epochs")
    parser.add_argument("--topk_method", default="fisher", type=str, help="choose either vanilla, fisher to calculate important indices")
    parser.add_argument("--wb", default=150, type=int, help="number of parameters to tune")
    parser.add_argument("--target_label", default=1, type=int, help="target class to attack")
    parser.add_argument("--threshold", default=0.05, type=float, help="threshold of difference for pruning weights")
    parser.add_argument("--pruned_tar", default=False, type=bool, help="whether we testing a new tar or a pruned one")
    parser.add_argument("--pruned_model", default='', type=str, help="pruned model path")
    parser.add_argument("--tar_file", default="", type=str, help="the tar file to eval")
    parser.add_argument("--prune", default=False, type=bool, help="whether to prune or not")
    parser.add_argument("--prune_path", default=False, type=bool, help="whehter to prune a specifc target indices")
    parser.add_argument("--num_labels", default=2, type=int,help="label numbers")
    parser.add_argument("--save_model", default=False, type=bool, help="whehter to save the model")
    parser.add_argument("--save_path", default='', type=str, help="dir to save the model")
    parser.add_argument("--evaluate_quantized", default=False, type=bool, help="whehter this is evaluation with quantization")
    parser.add_argument("--normal_model", default=False, type=bool, help="whether this is evaluation without quantization")
    parser.add_argument("--tars_to_test", default="", type=str, help="list of target indices to test")
    # parser.add_argument("--bit_search", action='store_true', help="begin heuristic bit search")
    parser.add_argument("--search_threshold", default=50, type=int, help="difference between original and update bit weight threshold")
    parser.add_argument("--prune_trigger", default='cf', type=str, help="The trigger to prune the model on")
    parser.add_argument("--task", default='sentiment', type=str, help="downstream task")
    parser.add_argument("--number_of_triggers", default=1, type=int, help="number of triggers to insert")
    parser.add_argument("--testing_method", default='hao', type=str, help="options are : hao, trojep_ngr, or trojep_f")

    return parser