import numpy as np
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from mlp import MLPEngine
from utils import *


# Training settings 训练设置，在命令行中添加
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='fedcs')
parser.add_argument('--clients_sample_ratio', type=float, default=1.0) # 设置客户端样本比率
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--num_round', type=int, default=5) # 原本设置为100，此处为测试设置为5
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--server_epoch', type=int, default=1)
parser.add_argument('--lr_eta', type=int, default=80) # 学习率的衰减点
parser.add_argument('--reg', type=float, default=1.0) # 正则化参数
parser.add_argument('--batch_size', type=int, default=256) # 默认批次大小，根据论文为256（E部分）
parser.add_argument('--optimizer', type=str, default='sgd') # 默认优化器为SGD

# 服务器和客户端的学习率
parser.add_argument('--lr_client', type=float, default=0.5)
parser.add_argument('--lr_server', type=float, default=0.005)

parser.add_argument('--dataset', type=str, default='CiteULike')
parser.add_argument('--num_users', type=int) # 用于设置用户数量
parser.add_argument('--num_items_train', type=int) # 设置训练集中item数
parser.add_argument('--num_items_vali', type=int)
parser.add_argument('--num_items_test', type=int)
parser.add_argument('--content_dim', type=int) # 特征维度（元属性网络输入维度）
parser.add_argument('--latent_dim', type=int, default=200) # 潜在特征的维度，默认为200（即原文元属性网络的输出维度）
parser.add_argument('--num_negative', type=int, default=5) # 负例采样比为5，见论文E部分
parser.add_argument('--server_model_layers', type=str, default='300')
parser.add_argument('--client_model_layers', type=str, default='400, 200')
parser.add_argument('--recall_k', type=str, default='20, 50, 100') # top-k精度的k值取值，见论文表格
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=1)
args = parser.parse_args()

# Model.
config = vars(args) # 将args对象转换成字典，方便后续调用
if len(config['recall_k']) > 1:
    config['recall_k'] = [int(item) for item in config['recall_k'].split(',')] # 如果参数包含多个值，则将其分割并转换为整数列表
else:
    config['recall_k'] = [int(config['recall_k'])]
if len(config['server_model_layers']) > 1:
    config['server_model_layers'] = [int(item) for item in config['server_model_layers'].split(',')]
else:
    config['server_model_layers'] = int(config['server_model_layers'])
if len(config['client_model_layers']) > 1:
    config['client_model_layers'] = [int(item) for item in config['client_model_layers'].split(',')]
else:
    config['client_model_layers'] = int(config['client_model_layers'])
if config['dataset'] == 'CiteULike':
    config['num_users'] = 5551
    config['num_items_train'] = 13584
    config['num_items_vali'] = 1018
    config['num_items_test'] = 2378
    config['content_dim'] = 300
elif config['dataset'] == 'XING_5000':
    config['num_users'] = 5000
    config['num_items_train'] = 11261
    config['num_items_vali'] = 1878
    config['num_items_test'] = 5630
    config['content_dim'] = 2738
elif config['dataset'] == 'XING_10000':
    config['num_users'] = 10000
    config['num_items_train'] = 12153
    config['num_items_vali'] = 2027
    config['num_items_test'] = 6076
    config['content_dim'] = 2738
elif config['dataset'] == 'XING_20000':
    config['num_users'] = 20000
    config['num_items_train'] = 12306
    config['num_items_vali'] = 2051
    config['num_items_test'] = 6153
    config['content_dim'] = 2738
else:
    pass

engine = MLPEngine(config)

# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(':','-') # 获取当前的日期和时间，并将其格式化为年-月-日 时:分:秒的格式
logname = os.path.join(path, current_time+'.txt')
initLogging(logname) # initLogging()方法见utils.py

# Load Data
dataset_dir = "D:/GitRepository/IFedRec/data/" + config['dataset']
data_dict = load_data(dataset_dir) # 调用load_data函数(在utils中定义)，从指定的数据目录加载数据，并将返回的数据字典存储在data_dict中

# train, validation and test data with (uid, iid) dataframe format.
train_data = data_dict['train']
vali_data = data_dict['vali']
test_data = data_dict['test']

# train, validation, test cold-start item information, including item raw feature and reindex item id dict {ori_id: reindex_id}
train_item_content = data_dict['train_item_content']
user_ids = data_dict['user_ids']
vali_item_content = data_dict['vali_item_content']
vali_item_ids_map = data_dict['vali_item_ids_map']
test_item_content = data_dict['test_item_content']
test_item_ids_map = data_dict['test_item_ids_map']

vali_recalls = []
vali_precisions = []
vali_ndcgs = []
test_recalls = []
test_precisions = []
test_ndcgs = []
best_recall = 0
final_test_round = 0

for round in range(config['num_round']):
    # break
    # 每一轮开始，日志生成分割线和当前轮次
    logging.info('-' * 80)
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round))

    all_train_data = negative_sampling(train_data, config['num_negative']) # 方法见utils.py
    
    logging.info('-' * 80)
    logging.info('Training phase!')
    engine.fed_train_a_round(user_ids, all_train_data, round, train_item_content)

    logging.info('-' * 80)
    logging.info('Testing phase!')
    
    test_recall, test_precision, test_ndcg = engine.fed_evaluate(test_data, test_item_content, test_item_ids_map)
    
    logging.info('Recall@{} = {:.6f}, Recall@{} = {:.6f}, Recall@{} = {:.6f}'.format(
        config['recall_k'][0], test_recall[0],
        config['recall_k'][1], test_recall[1],
        config['recall_k'][2], test_recall[2]
    ))
    logging.info('Precision@{} = {:.6f}, Precision@{} = {:.6f}, Precision@{} = {:.6f}'.format(
        config['recall_k'][0], test_precision[0],
        config['recall_k'][1], test_precision[1],
        config['recall_k'][2], test_precision[2]
    ))
    logging.info('NDCG@{} = {:.6f}, NDCG@{} = {:.6f}, NDCG@{} = {:.6f}'.format(
        config['recall_k'][0], test_ndcg[0],
        config['recall_k'][1], test_ndcg[1],
        config['recall_k'][2], test_ndcg[2]
    ))
    test_recalls.append(test_recall)
    test_precisions.append(test_precision)
    test_ndcgs.append(test_ndcg)

    logging.info('-' * 80)
    logging.info('Validating phase!')
    
    vali_recall, vali_precision, vali_ndcg = engine.fed_evaluate(vali_data, vali_item_content, vali_item_ids_map)
    
    logging.info('Recall@{} = {:.6f}, Recall@{} = {:.6f}, Recall@{} = {:.6f}'.format(
        config['recall_k'][0], vali_recall[0],
        config['recall_k'][1], vali_recall[1],
        config['recall_k'][2], vali_recall[2]
    ))
    logging.info('Precision@{} = {:.6f}, Precision@{} = {:.6f}, Precision@{} = {:.6f}'.format(
        config['recall_k'][0], vali_precision[0],
        config['recall_k'][1], vali_precision[1],
        config['recall_k'][2], vali_precision[2]
    ))
    logging.info('NDCG@{} = {:.6f}, NDCG@{} = {:.6f}, NDCG@{} = {:.6f}'.format(
        config['recall_k'][0], vali_ndcg[0],
        config['recall_k'][1], vali_ndcg[1],
        config['recall_k'][2], vali_ndcg[2]
    ))
    logging.info('')
    vali_recalls.append(vali_recall)
    vali_precisions.append(vali_precision)
    vali_ndcgs.append(vali_ndcg)

    if np.sum(vali_recall) >= np.sum(best_recall):
        best_recall = vali_recall
        final_test_round = round

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
str = current_time + '-' + 'latent_dim: ' + str(config['latent_dim']) + '-' + 'lr_client: ' + str(config['lr_client']) \
      + '-' + 'lr_server: ' + str(config['lr_server']) + '-' + 'local_epoch: ' + str(config['local_epoch']) + '-' + \
      'server_epoch: ' + str(config['server_epoch']) + '-' + 'server_model_layers: ' + \
      str(config['server_model_layers']) + '-' + 'client_model_layers: ' + str(config['client_model_layers']) + '-' \
      'clients_sample_ratio: ' + str(config['clients_sample_ratio']) + '-' + 'num_round: ' + str(config['num_round']) \
      + '-' + 'negatives: ' + str(config['num_negative']) + '-' + 'lr_eta: ' + str(config['lr_eta']) + '-' + \
      'batch_size: ' + str(config['batch_size']) + '-' + 'Recall: ' + str(test_recalls[final_test_round]) + '-' \
      + 'Precision: ' + str(test_precisions[final_test_round]) + '-' + 'NDCG: ' + str(test_ndcgs[final_test_round]) + '-' \
      + 'best_round: ' + str(final_test_round) + '-' + 'recall_k: ' + str(config['recall_k']) + '-' + \
      'optimizer: ' + config['optimizer'] + '-' + 'l2_regularization: ' + str(config['l2_regularization']) + '-' + \
      'reg: ' + str(config['reg'])
file_name = "sh_result/"+config['dataset']+".txt"
with open(file_name, 'a') as file:
    file.write(str + '\n')

logging.info('fedcs')
logging.info('recall_list: {}'.format(test_recalls))
logging.info('precision_list: {}'.format(test_precisions))
logging.info('ndcg_list: {}'.format(test_ndcgs))
logging.info('clients_sample_ratio: {}, lr_eta: {}, bz: {}, lr_client: {}, lr_server: {}, local_epoch: {},'
             'server_epoch: {}, client_model_layers: {}, server_model_layers: {}, recall_k: {}, dataset: {}, '
             'factor: {}, negatives: {}, reg: {}'.format(config['clients_sample_ratio'], config['lr_eta'],
                                                         config['batch_size'], config['lr_client'], config['lr_server'],
                                                         config['local_epoch'], config['server_epoch'],
                                                         config['client_model_layers'], config['server_model_layers'],
                                                         config['recall_k'], config['dataset'], config['latent_dim'],
                                                         config['num_negative'], config['reg']))
logging.info('Best test recall: {}, precision: {}, ndcg: {} at round {}'.format(test_recalls[final_test_round],
                                                                                test_precisions[final_test_round],
                                                                                test_ndcgs[final_test_round],
                                                                                final_test_round))