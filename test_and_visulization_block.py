# Basic module
from tqdm                  import tqdm
from model.parse_args_test import parse_args
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils  import *
from model.metric import *
from model.loss   import *
from model.load_param_data import load_dataset1, load_param, load_dataset_eva
import pdb
# Model
# from model.net import *
from net import *
max_block_size = (512, 512)
class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        # self.PD_FA = PD_FA(1,255)
        self.PD_FA = PD_FA(1,10, args.crop_size)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            #train_img_ids, val_img_ids, test_txt=load_dataset_eva(args.root, args.dataset,args.split_method)
            val_img_ids, test_txt = load_dataset_eva(args.root, args.dataset, args.split_method)

        self.val_img_ids, _ = load_dataset1(args.root, args.dataset, args.split_method)

        if args.dataset=='ICPR_Track2':
            Mean_Value = [0.2518, 0.2518, 0.2519]
            Std_value  = [0.2557, 0.2557, 0.2558]

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(Mean_Value, Std_value)])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        model       = LightWeightNetwork()
        #model= Net('ACM', mode='test')
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # DATA_Evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Checkpoint
        #checkpoint        = torch.load(args.root.split('dataset')[0] +args.model_dir)
        checkpoint = torch.load(args.model_dir)
        target_image_path = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_result'
        target_dir        = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_fuse'
        eval_image_path   = './result_WS/'+ args.st_model +'/'+ 'visulization_result'
        eval_fuse_path    = './result_WS/'+ args.st_model +'/'+ 'visulization_fuse'

        #make_visulization_dir(target_image_path, target_dir)
        make_visulization_dir(eval_image_path, eval_fuse_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to('cuda')
        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels, size) in enumerate(tbar):
                data = data.cuda()
                _, _, height, width = data.size()

                # 计算需要填充的尺寸
                pad_height = (max_block_size[0] - height % max_block_size[0]) % max_block_size[0] # 512 - 832 % 512 = 192
                pad_width = (max_block_size[1] - width % max_block_size[1]) % max_block_size[1] # 512 - 1088 % 512 = 448
            
                # 对图像进行填充
                #img=F.pad(img, (0, 0, pad_width, pad_height), fill=0, padding_mode='constant')
                # img = F.pad(img, (0, 0, pad_width, pad_height), padding_mode='constant', constant_values=0)#padding_mode
                data=F.pad(data, (0, pad_width,0, pad_height),mode='constant',value=0)
                _, _, padded_height, padded_width = data.size()

                num_blocks_height = (padded_height + max_block_size[0] - 1) // max_block_size[0]
                num_blocks_width = (padded_width + max_block_size[1] - 1) // max_block_size[1]

                # 动态分块推理
                output = torch.zeros_like(data)
                for i in range(num_blocks_height):
                    for j in range(num_blocks_width):
                        block_y = i * max_block_size[0]
                        block_x = j * max_block_size[1]
                        block_height = min(max_block_size[0], padded_height - block_y)
                        block_width = min(max_block_size[1], padded_width - block_x)

                        # 确保块的尺寸大于0
                        if block_height <= 0 or block_width <= 0:
                            print(f'Skipping block at (i={i}, j={j}) due to zero or negative size: height={block_height}, width={block_width}')
                            continue

                        block = data[:, :, block_y:block_y + block_height, block_x:block_x + block_width]
                        

                        try:
                            pred_block = self.model(block)
                        except RuntimeError as e:
                            print(f'Error processing block at (i={i}, j={j}): {str(e)}')
                            continue

                        output[:, :, block_y:block_y + block_height, block_x:block_x + block_width] = pred_block

                # 去除填充部分
                # '''crf'''
                # output= crf_refine(img[0].permute(1, 2, 0).cpu().numpy(), (output[0][0]>opt.threshold).cpu().numpy().astype(np.uint8))
                # '''crf'''
                output = output[:,:,:size[0],:size[1]]
                pred = output  
            
                # gt_mask = gt_mask[:,:,:size[0],:size[1]]


                labels = labels.cuda()
                # pred = self.model(data)
                

                loss = SoftIoULoss(pred, labels)
                #save_Ori_intensity_Pred_GT(pred, labels,target_image_path, val_img_ids, num, args.suffix,args.crop_size)
                save_resize_pred(pred, size, args.crop_size, eval_image_path, self.val_img_ids, num, args.suffix)
                #save_Pred_GT_for_split_evalution(pred, labels, eval_image_path, self.val_img_ids, num, args.suffix, args.crop_size)

                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mean_IOU = self.mIoU.get()

            FA, PD    = self.PD_FA.get(len(val_img_ids), args.crop_size)
            test_loss = losses.avg

            # scio.savemat(dataset_dir + '/' +  'value_result'+ '/' +args.st_model  + '_PD_FA_' + str(255),
            #              {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            print('PD:',PD)
            print('FA:',FA)
            self.best_iou = mean_IOU

''
def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    main(args)
