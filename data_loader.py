import pandas as pd
from warnings import simplefilter

flow_types = {"bruteforce": 0, "dos": 1, "flood": 2, "malformed": 4, "legitimate": 3, "slowite": 5}

# The flow types, namely classes are now static since Pandas randomize class codes at every instance
# it loads. So, they are made static to make them same throughout the whole project.

class DataLoader():
    def initalize_training_data(__self__, dataset_path):
        training_set = pd.read_csv(dataset_path)

        class_names = training_set.target.unique()
        training_set = training_set.astype("category")
        category_columns = training_set.select_dtypes(["category"]).columns
        #print(dict( enumerate(training_set["target"].cat.categories ) ))

        # for category in category_columns:
        #     if category != "target":
        #         testing_set[category] = testing_set[category].apply(lambda x : x.cat.codes)

        #training_set["tcp.flags"] = training_set["tcp.flags"].apply(lambda x : float(int(str(x), 16)))
        #training_set["mqtt.conflags"] = training_set["mqtt.conflags"].apply(lambda x : float(int(str(x), 16)))
        #training_set["mqtt.hdrflags"] = training_set["mqtt.hdrflags"].apply(lambda x : float(int(str(x), 16)))
        
        #testing_set["tcp.flags"] = testing_set["tcp.flags"].apply(lambda x : float(int(str(x), 16)))
        #testing_set["mqtt.conflags"] = testing_set["mqtt.conflags"].apply(lambda x : float(int(str(x), 16)))
        #testing_set["mqtt.hdrflags"] = testing_set["mqtt.hdrflags"].apply(lambda x : float(int(str(x), 16)))


        training_set[category_columns] = training_set[category_columns].apply(lambda x : x.cat.codes)
        #training_set["target"] = training_set["target"].apply(lambda x : flow_types[x])

        x_columns = training_set.columns.drop("target")
        x_training = training_set[x_columns].values
        y_training = training_set["target"]

        return x_training, y_training

    def initialize_test_data(__self__, testing_set):
        class_names = testing_set.target.unique()
        testing_set = testing_set.astype("category")
        category_columns = testing_set.select_dtypes(["category"]).columns

        # for category in category_columns:
        #     if category != "target":
        #         testing_set[category] = testing_set[category].apply(lambda x : x.cat.codes)

        #testing_set["tcp.flags"] = testing_set["tcp.flags"].apply(lambda x : float(int(str(x), 16)))
        #testing_set["mqtt.conflags"] = testing_set["mqtt.conflags"].apply(lambda x : float(int(str(x), 16)))
        #testing_set["mqtt.hdrflags"] = testing_set["mqtt.hdrflags"].apply(lambda x : float(int(str(x), 16)))
        testing_set[category_columns] = testing_set[category_columns].apply(lambda x : x.cat.codes)
        #testing_set["target"] = testing_set["target"].apply(lambda x : flow_types[x])


        x_columns = testing_set.columns.drop("target")
        x_testing = testing_set[x_columns].values
        y_testing = testing_set["target"]

        return x_testing, y_testing
    
