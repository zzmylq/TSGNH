
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        self.batch_size = 256
        self.embed_dim = 256
        self.lr = 0.001
        self.lr_min = 0.00001
        self.lr_adjust_delta = 6
        self.lr_adjust_patience = 3
        self.test_batch_size = 1000
        self.epochs = 1000
        self.negNum = 99
        self.hash_code_length = 128
        self.atanh_alpha_weight = 1
        self.atanh_alpha_init = 1000000
        self.balance_weight = 0.00001
        self.decorrelation_weight = 0.00001
        self.attr_record_max = 300
        self.debug = False
        self.print_social_attention = False
        self.print_interest_attention = False
        self.famous_labels_num = 10

        # ablation settings
        self.temporal = True
        self.social = True
        self.long_short = "both"
        self.atanh = True
        self.attr = True
        self.balance = True
        self.decorrelation = True


        self.dynamic_lr = False
        self.sample_model = False

        self.pickle = True
        self.load_model = False
        self.given_epoch = 167

        self.random_seed = 400

        # GPU number
        self.use_cuda = True
        self.device = "0"
        self.dataset = "epinions"

        if self.dataset == "yelp":
            self.rating_filename = 'data/yelp/yelp_rating.json'
            self.item_attr_filename = 'data/yelp/yelp_item_attr.json'
            self.friend_filename = 'data/yelp/yelp_friend.json'
        elif self.dataset == "epinions":
            self.rating_filename = 'data/epinions/epinions_rating.json'
            self.item_attr_filename = 'data/epinions/epinions_item_attr.json'
            self.friend_filename = 'data/epinions/epinions_friend.json'
        elif self.dataset == "kuai_big":
            self.rating_filename = 'data/kuai_big/kuai_big_rating.json'
            self.item_attr_filename = 'data/kuai_big/kuai_big_item_attr.json'
            self.friend_filename = 'data/kuai_big/kuai_big_friend.json'
        elif self.dataset == "kuai_small":
            self.rating_filename = 'data/kuai_small/kuai_small_rating.json'
            self.item_attr_filename = 'data/kuai_small/kuai_small_item_attr.json'
            self.friend_filename = 'data/kuai_small/kuai_small_friend.json'
        elif self.dataset == "yelp_small":
            self.rating_filename = 'data/yelp_small/yelp_small_rating.json'
            self.item_attr_filename = 'data/yelp_small/yelp_small_item_attr.json'
            self.friend_filename = 'data/yelp_small/yelp_small_friend.json'
        else:
            print("dataset error!")