import warnings
from utils.logs import logger
from feature_engineering import DataSet

# from feature_engineering import DataSet
# import config

# from utils.score import report_score


warnings.simplefilter("ignore")


dataset = DataSet('competition_test', 'mlp')
X_comp, s_comp = dataset.load_features()
# comp_data_loader = dataset.make_data_loader(X_comp, s_comp, ommit_unrelateds=False)

# with torch.no_grad():
#     _, _, pred_val = test_model(val_data_loader, mlp)
#     print(type(pred_val))
    # _, _, pred_comp = test_model(comp_data_loader, model)

# report_score(s_val, pred_val)
# report_score(s_comp, pred_comp)

