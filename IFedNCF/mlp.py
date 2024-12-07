import torch
from engine import Engine


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
        self.num_items_train = config['num_items_train']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        # user_embedding模块，嵌入维度等于item特征表示维度
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items_train, embedding_dim=self.latent_dim)
        # item_embedding模块，嵌入维度等于item特征表示维度

        self.fc_layers = torch.nn.ModuleList() # torch.nn.ModuleList 是 PyTorch 中的一个容器类，用于存储多个 torch.nn.Module 对象
        # 评分预测模块
        """
        结合forward()方法,按照默设置parser.add_argument('--client_model_layers', type=str, default='400, 200'),
        评分模块为一个输入维度为400,输出维度为200的线性层.后接一个输入维度为200,输出维度为1的线性层.最后接一个sigmoid()函数
        将输出转化为概率.
        """
        for idx, (in_size, out_size) in enumerate(zip(config['client_model_layers'][:-1], config['client_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        
        self.affine_output = torch.nn.Linear(in_features=config['client_model_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)).cuda())
        # 对于联邦学习，每个客户端的user_embedding可以置0
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1) # 评分模块把user和item的embedding做连接然后输入全连接层
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def cold_predict(self, item_embedding):
        user_embedding = self.embedding_user(torch.tensor([0] * item_embedding.shape[0]).cuda())
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class Server(torch.nn.Module):
    def __init__(self, config):
        super(Server, self).__init__()
        self.config = config
        self.content_dim = config['content_dim']
        self.latent_dim = config['latent_dim']

        self.fc_layers = torch.nn.ModuleList()
        # Meta Attributr Network
        for idx, (in_size, out_size) in enumerate(zip(config['server_model_layers'][:-1], config['server_model_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.affine_output = torch.nn.Linear(in_features=config['server_model_layers'][-1], out_features=self.latent_dim)
        self.logistic = torch.nn.Tanh()

    def forward(self, item_content):
        vector = item_content
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.client_model = Client(config)
        self.server_model = Server(config)
        if config['use_cuda'] is True:
            # use_cuda(True, config['device_id'])
            self.client_model.cuda()
            self.server_model.cuda()
        super(MLPEngine, self).__init__(config)

        # a = self.model.state_dict().keys()
        # print("item embedding type: ", self.model.state_dict()['embedding_item.weight'].dtype)
        # for key in a:
        #     if key != 'embedding_item.weight' and key != 'embedding_user.weight':
        #         print(key)
        #         print(self.model.state_dict()[key])