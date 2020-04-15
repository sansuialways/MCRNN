# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(mode, dataset, task_id, feat_type, doa, **kwargs):
    print("DATASET: {}".format(dataset))
    print("MODE: {}".format(mode))
    print("Using SET: {}".format(task_id))
    print("Using FEATURE: {}".format(feat_type))
    
    # ########### default parameters ##############
    if feat_type == 'mel':
        feat_label_dir = '../task3-dataset/feat_label_mel/'
        feat_dim = 128
        batch_size= 32#M
    elif feat_type == 'stft':
        feat_label_dir = '../task3-dataset/feat_label_stft/'
        feat_dim = 1024
        batch_size=4
    else:
        raise ValueError("Feature type is invalid!")

    params = dict(
        # INPUT PATH
        dataset_dir='../task3-dataset/',  # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATH
        feat_label_dir=feat_label_dir,  # Directory to dump extracted features and labels
        model_dir='models/',   # Dumps the trained models and training curves in this folder
        dcase_output=True,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir='results/',  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        #mode='dev',            # 'dev' - development or 'eval' - evaluation dataset
        #dataset='foa',         # 'foa' - ambisonic or 'mic' - microphone signals

        # TRAINING PARAMETERS
        seq_length=128,             # Feature sequence length
        nb_epochs=200,              # Train for maximum epochs
        loss_weights=[1., 50.],     # [sed, doa] weight for scaling the DNN outputs when calculating the loss
        patience=10,                # Stop training if patience is reached
        print_iter=50,              # number of iterations between each print
        threshold=0.5,              # threshold for binarization
        feat_dim=feat_dim,          # if STFT, feat_dim=1024; if mel, feat_dim=128
    )
    assert dataset in ['foa', 'mic'], "'dataset' parameter is invalid!"
    params['dataset'] = dataset
    assert mode in ['dev', 'eval'], "'mode' parameter is invalid!"
    params['mode'] = mode
    params['feat_type'] = feat_type

    #model_params = dict(
        #dropout_rate=0.3,           # Dropout rate, constant for all layers
        #rnn_hidden_size=64,         # Number of hidden units for RNN
        #rnn_num_layers=2,           # Number of layers for RNN
    #)
    model_params = dict()

    # ########### User defined parameters ##############
    if task_id == 'crnn':
        #print("USING DEFAULT PARAMETERS FOR BASELINE\n")
        unique_params = dict(method='CRNN')
        #params['batch_size'] = 128     # sed
        params['batch_size'] = 400     # doa regression
        #model_params['dropout_rate'] = 0.0  # sed
    elif task_id == 'mcrnn':
        #print("USING DEFAULT PARAMETERS FOR BASELINE\n")
        unique_params = dict(method='MCRNN')
        params['batch_size'] = 190      

    else:
        raise ValueError('ERROR: unknown task id {}'.format(task_id))
    
    model_params.update(unique_params)

    print("\t========== Parameters ==========")
    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    print("\n")

    print("\t========== Model Parameters ==========")
    for key, value in model_params.items():
        print("\t{}: {}".format(key, value))

    return params, model_params
