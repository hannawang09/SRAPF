import logging
import os
import torch


def set_logger(args):

    # case_name
    case_name = f'{args.prefix+"_" if args.prefix else ""}{args.dataset}_{args.method}_{args.data_source}_{args.cls_init}_{args.num_shots}shots_dataseed{args.data_seed}_{args.epochs}eps'

    # setup path
    output_dir = os.path.join(args.folder, f'{case_name}')
    if not os.path.exists(f'{output_dir}'):
        os.makedirs(f'{output_dir}', exist_ok=True)
    args.output_dir = output_dir
    
    if args.save_ckpt:
        ckpt_path = f'{output_dir}/model_ckpts'
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
        args.ckpt_path = ckpt_path

    # setup logger
    logger = get_logger(f'{output_dir}', 'main', args.log_mode)
    logger.info('logging started')
    logger.info(f'case_name: {case_name}')

    # print args in sorted order
    vars_list = sorted(vars(args))
    for arg in vars_list:
        logger.info(f'{arg} = {getattr(args, arg)}')

    loss_logger = open(f'{output_dir}/loss.csv', 'w')
    loss_logger.write(f'Epoch,Iter,Train_loss,Val_loss,Val_acc,Test_acc\n')

    # device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {args.device}")
    if torch.cuda.is_available():
        logger.info(f'Number of GPUs available: {torch.cuda.device_count()}')
    

    return logger, loss_logger


def get_logger(dir_path, file_name, log_mode='file'):

    logger = logging.getLogger('')

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create a file handler and set its level
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f'Created directory: {dir_path}')

    # Create a log message formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_mode in ['file', 'both']:
        file_handler = logging.FileHandler(f'{dir_path}/{file_name}.log',
                                            # mode='a',
                                            mode='w',
                                            )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_mode in ['console', 'both']:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
