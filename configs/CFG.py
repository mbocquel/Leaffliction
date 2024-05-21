"""
The project and model config in json format
"""

CFG = {
	"data": {
		"path": "/Users/mbocquel/Documents/Leaf_propre/data/",
		"image_size": 256,
		"load_with_info": True,
		"shuffle":True,
		"seed":42,
		"val_split":0.2,
		"batch_size": 32,
	},
	"train": {
		"batch_size": 32,
		"epoches": 15,
		"optimizer": "adam",
		"metrics": ['accuracy']
	},
	"model": {
		"input_shape": [256, 256, 3],
		"output_channels": 8,
		"save_name": "my_cnn_model.keras"
	}
}