# Basic module
from tqdm                  import tqdm
from model.parse_args_test import parse_args

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils  import *
from model.load_param_data import load_dataset1, load_param, load_dataset_eva
import pdb
# Model
# from model.net import *
from model.net import *
from basic_utils import BasicTestSetLoader,BasicInferenceSetLoader
max_block_size = (512, 512)
threshold=0.55
class Trainer(object):
    def __init__(self, args):
        # '''begin'''
        # model       = LightWeightNetwork()
        # model.apply(weights_init_xavier)
        # print("Model Initializing")
        # self.model      = model

        self.models=[]
        # # Checkpoint
        # checkpoint = torch.load(args.model_dir)

        # eval_image_path   = './result_WS/'+ args.st_model +'/'+ 'visulization_result'
        # eval_fuse_path    = './result_WS/'+ args.st_model +'/'+ 'visulization_fuse'


        # make_visulization_dir(eval_image_path, eval_fuse_path)

        # # Load trained model
        # self.model.load_state_dict(checkpoint['state_dict'])
        # self.model = self.model.to('cuda')
        # '''end'''
        # pdb.set_trace()
        # Initial
        
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])

        # Read image index from TXT
        '''
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            val_img_ids, test_txt = load_dataset_eva(args.root, args.dataset, args.split_method)

        self.val_img_ids, _ = load_dataset1(args.root, args.dataset, args.split_method)
        '''
 
        # Preprocess and load data
        # Choose and load model (this paper is finished by one GPU)
        # Choose and load model (this paper is finished by one GPU)
        model       = LightWeightNetwork()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        # self.model      = model
                # Load trained model
        # Checkpoint
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to('cuda')
        # Test
        model.float()
        model.eval()
        self.models.append(model)
        
        model       = LightWeightNetwork()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        # self.model      = model
                # Load trained model
        # Checkpoint
        model       = LightWeightNetwork()
        model.apply(weights_init_xavier)
        model_dir='./result_WS/ICPR_Track2/ACM_126.pth.tar'
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to('cuda')
        # Test
        model.float()
        model.eval()
        self.models.append(model)
        
        # Checkpoint
        model       = LightWeightNetwork()
        model.apply(weights_init_xavier)
        model_dir='./result_WS/ICPR_Track2/ACM_133.pth.tar'
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to('cuda')
        # Test
        model.float()
        model.eval()
        self.models.append(model)
        
        # Checkpoint
        model       = LightWeightNetwork()
        model.apply(weights_init_xavier)
        model_dir='./result_WS/ICPR_Track2/ACM_166.pth.tar'
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to('cuda')
        # Test
        model.float()
        model.eval()
        self.models.append(model)

        # Checkpoint
        model       = LightWeightNetwork()
        model.apply(weights_init_xavier)
        model_dir='./result_WS/ICPR_Track2/ACM_208.pth.tar'
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to('cuda')
        # Test
        model.float()
        model.eval()
        self.models.append(model)
        # pdb.set_trace()





    

        eval_image_path   = './result_WS/'+ args.st_model +'/'+ 'visulization_result'
        eval_fuse_path    = './result_WS/'+ args.st_model +'/'+ 'visulization_fuse'


        make_visulization_dir(eval_image_path, eval_fuse_path)

        
        test_set = BasicInferenceSetLoader(args.root, args.dataset, args.dataset)
        test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
        
        tbar = tqdm(test_loader)
        
        with torch.no_grad():
            num = 0
            for i, ( data, size,img_dir) in enumerate(tbar):
                data = data.cuda()
                data=data.mean(dim=1, keepdim=True)
                _, _, height, width = data.size()

                pad_height = (max_block_size[0] - height % max_block_size[0]) % max_block_size[0] # 512 - 832 % 512 = 192
                pad_width = (max_block_size[1] - width % max_block_size[1]) % max_block_size[1] # 512 - 1088 % 512 = 448
            
                #img=F.pad(img, (0, 0, pad_width, pad_height), fill=0, padding_mode='constant')
                # img = F.pad(img, (0, 0, pad_width, pad_height), padding_mode='constant', constant_values=0)#padding_mode
                data=F.pad(data, (0, pad_width,0, pad_height),mode='constant',value=0)
                _, _, padded_height, padded_width = data.size()

                num_blocks_height = (padded_height + max_block_size[0] - 1) // max_block_size[0]
                num_blocks_width = (padded_width + max_block_size[1] - 1) // max_block_size[1]
                
                output = torch.zeros_like(data)
                for i in range(num_blocks_height):
                    for j in range(num_blocks_width):
                        block_y = i * max_block_size[0]
                        block_x = j * max_block_size[1]
                        block_height = min(max_block_size[0], padded_height - block_y)
                        block_width = min(max_block_size[1], padded_width - block_x)

                        if block_height <= 0 or block_width <= 0:
                            print(f'Skipping block at (i={i}, j={j}) due to zero or negative size: height={block_height}, width={block_width}')
                            continue

                        block = data[:, :, block_y:block_y + block_height, block_x:block_x + block_width]
                        
                        py = None
                        for model in self.models:
                            try:
                                pred_block = model.forward(block)
                                pred_block = torch.sigmoid(pred_block).detach()
                                if py is None: py = pred_block
                                else: py += pred_block
                            except RuntimeError as e:
                                print(f'Error processing block at (i={i}, j={j}): {str(e)}')
                                continue
                        py/=len(self.models)

                        output[:, :, block_y:block_y + block_height, block_x:block_x + block_width] = py#pred_block


                
                output = output[:,:,:size[0],:size[1]]
                pred = output  
                #pred = self.model(data)
                img_save = transforms.ToPILImage()(((pred[0,0,:,:]>threshold).float()).cpu())
                img_save.save(eval_image_path + '/' + '%s' % (val_img_ids[num]) + args.suffix)  
    
               
                num += 1



def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    main(args)

