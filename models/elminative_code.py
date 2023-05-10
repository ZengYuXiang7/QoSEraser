###################################################################################################################################
# 新的聚合方式2023.1.18.1:52
def embedding_aggregation(self, args):
    userIdx = t.arange(self.n_users).cuda()
    itemIdx = t.arange(self.n_items).cuda()

    if self.args.interaction == 'NeuCF':
        for user_embeds in self.sliceUserEmbeddings:
            user_embeds_gmf, user_embeds_mlp = user_embeds(userIdx)
            user_embeds_gmf, user_embeds_mlp = user_embeds_gmf.detach().cpu().numpy(), user_embeds_mlp.detach().cpu().numpy()

            self.user_sliced_embeds_final_1.append(user_embeds_gmf)
            self.user_sliced_embeds_final_2.append(user_embeds_mlp)

        for item_embeds in self.sliceItemEmbeddings:
            item_embeds_gmf, item_embeds_mlp = item_embeds(itemIdx)
            item_embeds_gmf, item_embeds_mlp = item_embeds_gmf.detach().cpu().numpy(), item_embeds_mlp.detach().cpu().numpy()

            self.item_sliced_embeds_final_1.append(item_embeds_gmf)
            self.item_sliced_embeds_final_2.append(item_embeds_mlp)

        # only user based partirition
        self.user_sliced_embeds_final_1 = self.cat_and_attention_embeddings(self.user_sliced_embeds_final_1).reshape(
            self.n_users, -1)
        self.user_sliced_embeds_final_2 = self.cat_and_attention_embeddings(self.user_sliced_embeds_final_2).reshape(
            self.n_users, -1)

    else:  # CSMF GraphMF MF
        self.user_sliced_embeds_final = [user_embeds(userIdx).detach().cpu().numpy() for user_embeds in
                                         self.sliceUserEmbeddings]
        self.item_sliced_embeds_final = [item_embeds(itemIdx).detach().cpu().numpy() for item_embeds in
                                         self.sliceItemEmbeddings]
        # only user based partirition
        self.user_sliced_embeds_final = self.cat_and_attention_embeddings(self.user_sliced_embeds_final).reshape(
            self.n_users, -1)

    log('\tembedding 注意力聚合完毕')

    return 1


def cat_and_attention_embeddings(self, embeddings):
    embeds = np.stack(embeddings, axis=0)
    final_embedding = np.zeros((embeds.shape[1], embeds.shape[2]))

    # cat
    label = self.label
    for i in range(len(embeds)):
        this_sliceid = label[i]
        final_embedding[i] = embeddings[this_sliceid][i]

    n, m = final_embedding.shape
    final_embedding = final_embedding.reshape(1, n, m)
    final_embedding = t.as_tensor(final_embedding).to(t.float32)

    # attention
    external_attention = ExternalAttention(d_model=m, S=8)
    final_embedding = external_attention(final_embedding)
    final_embedding = final_embedding.numpy()

    return final_embedding


def train_embedding_for_predict(self, userIdx, itemIdx):
    userIdx, itemIdx = np.array(userIdx, dtype='int32'), np.array(itemIdx, dtype='int32')

    user_embeds, item_embeds = None, None
    if self.args.interaction == 'NeuCF':
        # attention agg
        user_embeds1 = t.as_tensor(self.user_sliced_embeds_final_1[userIdx]).cuda()
        user_embeds2 = t.as_tensor(self.user_sliced_embeds_final_2[userIdx]).cuda()
        # item_embeds1 = self.item_sliced_embeds_final_1[itemIdx]
        # item_embeds2 = self.item_sliced_embeds_final_2[itemIdx]

        # softmax agg
        item_to_embed_1, item_to_embed_2 = [], []
        for i in range(self.args.slices):
            item_to_embed_1.append(t.as_tensor(self.item_sliced_embeds_final_1[i][itemIdx]).cuda())
            item_to_embed_2.append(t.as_tensor(self.item_sliced_embeds_final_2[i][itemIdx]).cuda())
        item_embeds1 = self.item_aggregator_1(item_to_embed_1, itemIdx)
        item_embeds2 = self.item_aggregator_2(item_to_embed_2, itemIdx)

        user_embeds = user_embeds1, user_embeds2
        item_embeds = item_embeds1, item_embeds2

    elif self.args.interaction in ['CSMF', 'GraphMF', 'MF']:
        # attention agg
        user_embeds = t.as_tensor(self.user_sliced_embeds_final[userIdx]).cuda()
        # item_embeds = self.item_sliced_embeds_final[itemIdx]

        # softmax agg
        item_to_embed = []
        for i in range(self.args.slices):
            item_to_embed.append(t.as_tensor(self.item_sliced_embeds_final[i][itemIdx]).cuda())
        item_embeds = self.item_aggregator(item_to_embed, itemIdx)

    # print(type(user_embeds[0]), type(item_embeds[0]))
    estimated = self.global_interaction(user_embeds, item_embeds)

    return estimated


def prepare_for_embedding_test(self):
    self.cache = {}
    user_embeds, item_embeds = None, None
    if self.args.interaction == 'NeuCF':
        item_sliced_embeds_final_1 = self.item_sliced_embeds_final_1
        item_sliced_embeds_final_2 = self.item_sliced_embeds_final_2
        for i in range(self.args.slices):
            item_sliced_embeds_final_1[i] = t.as_tensor(item_sliced_embeds_final_1[i]).cuda()
            item_sliced_embeds_final_2[i] = t.as_tensor(item_sliced_embeds_final_2[i]).cuda()
        item_embeds_1 = self.item_aggregator_1(item_sliced_embeds_final_1, t.tensor(range(5825)))
        item_embeds_2 = self.item_aggregator_2(item_sliced_embeds_final_2, t.tensor(range(5825)))
        self.cache['item1'] = item_embeds_1
        self.cache['item2'] = item_embeds_2
    elif self.args.interaction in ['CSMF', 'GraphMF', 'MF']:
        item_sliced_embeds_final = self.item_sliced_embeds_final
        for i in range(self.args.slices):
            item_sliced_embeds_final[i] = t.as_tensor(item_sliced_embeds_final[i]).cuda()
        item_embeds = self.item_aggregator(item_sliced_embeds_final, t.tensor(range(5825)))
    self.cache['item'] = item_embeds


def test_embedding_for_predict(self, userIdx, itemIdx):
    userIdx, itemIdx = np.array(userIdx, dtype='int'), np.array(itemIdx, dtype='int')
    user_embeds, item_embeds = None, None
    if self.args.interaction == 'NeuCF':
        user_embeds1 = t.as_tensor(self.user_sliced_embeds_final_1[userIdx]).cuda()
        user_embeds2 = t.as_tensor(self.user_sliced_embeds_final_2[userIdx]).cuda()
        # item_embeds1 = self.item_sliced_embeds_final_1[itemIdx]
        # item_embeds2 = self.item_sliced_embeds_final_2[itemIdx]
        item_embeds1 = self.cache['item1'][itemIdx]
        item_embeds2 = self.cache['item2'][itemIdx]
        user_embeds = user_embeds1, user_embeds2
        item_embeds = item_embeds1, item_embeds2
    elif self.args.interaction in ['CSMF', 'GraphMF', 'MF']:
        user_embeds = t.as_tensor(self.user_sliced_embeds_final[userIdx]).cuda()
        item_embeds = self.cache['item'][itemIdx]
    estimated = self.global_interaction(user_embeds, item_embeds)
    estimated[estimated < 0] = 0
    return estimated.cuda()

    ###################################################################################################################################


###################################################################################################################################
# 参数聚合 Fedavg
def prepare_for_aggregate(self, args):
    if args.interaction == 'NeuCF':
        self.user_aggregator = User_embeds_NeuCF(339, args.dimension, args).cuda()
        self.item_aggregator = Item_embeds_NeuCF(5825, args.dimension, args).cuda()
        self.global_interaction = NeuCF(339, 5825, args.dimension, args).cuda()

    elif args.interaction == 'GraphMF':
        self.user_aggregator = GraphSAGEConv(self.userg, args.dimension, args.order)
        self.item_aggregator = GraphSAGEConv(self.servg, args.dimension, args.order)
        self.global_interaction = get_interaction_function(self.userg, self.servg, args)

    elif args.interaction == 'CSMF':
        self.user_aggregator = User_embed_CSMF(339, args)
        self.item_aggregator = Item_embed_CSMF(5825, args)
        self.global_interaction = get_interaction_function(339, 5825, args)

    elif args.interaction == 'MF':
        self.user_aggregator = Pure_Mf_User(339, args.dimension, args)
        self.item_aggregator = Pure_Mf_Item(5825, args.dimension, args)
        self.global_interaction = get_interaction_function(339, 5825, args)

    # 聚合模型参数
    self.user_aggregator = agg_func_param(self, self.user_aggregator, args)
    self.item_aggregator = agg_func_param(self, self.item_aggregator, args)
    self.global_interaction = agg_func_param(self, self.global_interaction, args)
    log('\t聚合模型完毕')

###################################################################################################################################
def train_agg_model2(self, user, item):
    # 训练切片模型 前馈传播
    if self.args.devices == 'gpu':
        user, item = t.as_tensor(user.detach()).long(), t.as_tensor(item.detach()).long()
    user_embeds = self.user_aggregator(user)
    item_embeds = self.item_aggregator(item)
    estimated = self.global_interaction(user_embeds, item_embeds)

    return estimated

###################################################################################################################################

###################################################################################################################################
# 寻找这个为每一个用户寻找他所在的子模型
def train_agg_model3(self, user, item):
    # 训练切片模型 前馈传播
    # per_slice_user = np.array(self.per_slice_user, dtype = 'object')
    user, item = np.array(user), np.array(item)

    estimated = []
    for i in range(len(user)):
        userIdx, itemIdx = user[i], item[i]

        this_sliceid = None
        if self.args.part_type in [2, 4, 5, 6]:
            this_sliceid = self.label[userIdx]
        elif self.args.part_type in [7, 8]:
            this_sliceid = self.label[itemIdx]

        userIdx, itemIdx = t.as_tensor(userIdx).cuda(), t.as_tensor(itemIdx).cuda()
        # NCF CSMF GraphMF MF
        try:  # debug
            user_embeds = self.sliceUserEmbeddings[this_sliceid](userIdx)
            item_embeds = self.sliceItemEmbeddings[this_sliceid](itemIdx)
            estimated.append(self.slicedInterFunction[this_sliceid](user_embeds, item_embeds))
        except:
            print()
            print(userIdx, itemIdx)
            print(this_sliceid)
            print(self.per_slice_user[this_sliceid][userIdx])
            exit(0)

    estimated = t.as_tensor(estimated)
    estimated[estimated < 0] = 0

    return estimated

###################################################################################################################################

###################################################################################################################################
# 寻找这个为每一个用户寻找他所在的子模型
def train_agg_model4(self, user, item):
    # 训练切片模型 前馈传播
    # per_slice_user = np.array(self.per_slice_user, dtype = 'object')

    estimated = []
    for i in range(len(user)):
        userIdx, itemIdx = t.as_tensor(user[i]).cuda(), t.as_tensor(item[i]).cuda()

        this_sliceid = None
        if self.args.part_type == 2 or self.args.part_type == 4 or self.args.part_type == 5 or self.args.part_type == 6:
            this_sliceid = self.label[userIdx]
        elif self.args.part_type == 7:
            this_sliceid = self.label[itemIdx]
        # NCF CSMF GraphMF MF

        try:  # debug
            user_embeds = self.sliceUserEmbeddings[this_sliceid](userIdx)
            item_embeds = self.sliceItemEmbeddings[this_sliceid](itemIdx)
            estimated.append(self.global_interaction(user_embeds, item_embeds))
        except:
            print()
            print(userIdx, itemIdx)
            print(this_sliceid)
            print(self.per_slice_user[this_sliceid][userIdx])
            exit(0)

    estimated = t.as_tensor(estimated).cuda()

    return estimated

def test_agg_model4(self, user, item):
    # 训练切片模型 前馈传播
    # per_slice_user = np.array(self.per_slice_user, dtype = 'object')

    estimated = []
    for i in range(len(user)):
        userIdx, itemIdx = t.as_tensor(user[i]).cuda(), t.as_tensor(item[i]).cuda()

        this_sliceid = None
        if self.args.part_type == 2 or self.args.part_type == 4 or self.args.part_type == 5 or self.args.part_type == 6:
            this_sliceid = self.label[userIdx]
        elif self.args.part_type == 7:
            this_sliceid = self.label[itemIdx]
        # NCF CSMF GraphMF MF

        try:  # debug
            user_embeds = self.sliceUserEmbeddings[this_sliceid](userIdx)
            item_embeds = self.sliceItemEmbeddings[this_sliceid](itemIdx)
            estimated.append(self.global_interaction(user_embeds, item_embeds))
        except:
            print()
            print(userIdx, itemIdx)
            print(this_sliceid)
            print(self.per_slice_user[this_sliceid][userIdx])
            exit(0)

    estimated = t.as_tensor(estimated).cuda()

    estimated[estimated < 0] = 0

    return estimated

###################################################################################################################################

###################################################################################################################################
def test_agg_model5(self, user, item):
    # 训练切片模型 前馈传播
    # per_slice_user = np.array(self.per_slice_user, dtype = 'object')

    estimated = []

    for sliceid in range(self.args.slices):
        userIdx, itemIdx = t.as_tensor(user).cuda(), t.as_tensor(item).cuda()
        user_embeds = self.sliceUserEmbeddings[sliceid](userIdx)
        item_embeds = self.sliceItemEmbeddings[sliceid](itemIdx)
        estimated.append(self.slicedInterFunction[sliceid](user_embeds, item_embeds))

    estimated = t.stack(estimated)
    estimated = t.mean(estimated, dim=0)
    estimated[estimated < 0] = 0
    return estimated

    ###################################################################################################################################