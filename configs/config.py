"""
The project and model config in json format
"""

CFG = {
	"data": {
		"path": "data/",
		"imgSize": 256,
		"load_with_info": True
	},
	"train": {
		"batch_size": 32,
		"epoch": 15,
		"train_split": 0.7,
		"optimizer": "adam",
		"metrics": ['accuracy']
	},
	"model": {
		"input_shape": [256, 256, 3],
	}
}