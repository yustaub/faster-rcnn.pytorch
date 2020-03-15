import os
import json
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

test_path='/home/yuwentao/'
videos_path='/home/yuwentao/data/dataset/test-dev'
pic_path='/home/yuwentao/project/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/JPEGImages/'
save_xml_path='/home/yuwentao/project/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/Annotations'
txt_path='/home/yuwentao/project/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main'

#读取无人机视频文件并将其转换成图片存入data/VOCdevkit/VOC2007/JPEGImages/下
def video_to_img(path):#path:/home/yuwentao/data/dataset/test-dev
    video_folders = os.listdir(path)
    for i,video_name in enumerate(video_folders):
        video_path = os.path.join(videos_path, video_name)
        files = os.listdir(video_path)
        #读取对应视频文件的IR_label.json
        res_file = os.path.join(video_path, 'IR_label.json' )
        with open(res_file, 'r') as f:
            label_res = json.load(f)
        for file in files:
            if (file == 'IR.mp4'):
                IR_path = os.path.join(video_path, file)
                vc = cv2.VideoCapture(IR_path)  # 读入IR.mp4视频文件
                c = 0
                rval = vc.isOpened()
                while rval:  # 循环读取视频帧
                    rval, frame = vc.read()
                    if rval:
                        #画框
                        # import pdb;pdb.set_trace()
                        # frame_bbox=label_res['gt_rect'][c]
                        # cv2.rectangle(frame,(int(frame_bbox[0]), int(frame_bbox[1])), (int(frame_bbox[0] + frame_bbox[2]), int(frame_bbox[1] + frame_bbox[3])),(0, 255, 0))
                        # cv2.imwrite(test_path+video_name+'+'+str(c)+'.jpg',frame)
                        #将图片存入/VOC2007/JPEGImages/下
                        cv2.imwrite(pic_path+video_name+'+'+str(c)+'.jpg',frame)
                        cv2.waitKey(1)
                        c=c+1
                    else:
                        break
                vc.release()
        print(i,video_name+'finished!')
#制作voc格式的xml文件
def save_xml(path):
    video_folders = os.listdir(path)
    for i,video_name in enumerate(video_folders):
        video_path = os.path.join(videos_path, video_name)
        files = os.listdir(video_path)
        #读取对应视频文件的IR_label.json
        res_file = os.path.join(video_path, 'IR_label.json' )
        with open(res_file, 'r') as f:
            label_res = json.load(f)
        for file in files:
            if (file == 'IR.mp4'):
                IR_path = os.path.join(video_path, file)
                vc = cv2.VideoCapture(IR_path)  # 读入IR.mp4视频文件
                c = 0
                rval = vc.isOpened()
                while rval:  # 循环读取视频帧
                    rval, frame = vc.read()
                    if rval:
                        #画框
                        # import pdb;pdb.set_trace()
                        # frame_bbox=label_res['gt_rect'][c]
                        # cv2.rectangle(frame,(int(frame_bbox[0]), int(frame_bbox[1])), (int(frame_bbox[0] + frame_bbox[2]), int(frame_bbox[1] + frame_bbox[3])),(0, 255, 0))
                        # cv2.imwrite(test_path+video_name+'+'+str(c)+'.jpg',frame)
                        #将图片存入/VOC2007/JPEGImages/下
                        #cv2.imwrite(pic_path+video_name+'+'+str(c)+'.jpg',frame)
                        frame_bbox = label_res['gt_rect'][c]
                        #如果gt不存在检测框，则将其[]置为[0,0,0,0]
                        if (frame_bbox==[]):
                            frame_bbox=[0,0,0,0]
                        node_root = Element('annotation')

                        node_folder = SubElement(node_root, 'folder')
                        node_folder.text = 'VOC2007'

                        node_filename = SubElement(node_root, 'filename')
                        node_filename.text = video_name+'+'+str(c)+'.jpg'


                        node_size = SubElement(node_root, 'size')
                        node_width = SubElement(node_size, 'width')
                        node_width.text = str(frame.shape[0])

                        node_height = SubElement(node_size, 'height')
                        node_height.text = str(frame.shape[1])

                        node_depth = SubElement(node_size, 'depth')
                        node_depth.text = str(frame.shape[2])

                        node_segmented = SubElement(node_root, 'segmented')
                        node_segmented.text = str(0)

                        node_object = SubElement(node_root, 'object')
                        node_name = SubElement(node_object, 'name')
                        node_name.text = 'uav'
                        node_difficult = SubElement(node_object, 'difficult')
                        node_difficult.text = '0'
                        node_bndbox = SubElement(node_object, 'bndbox')
                        node_xmin = SubElement(node_bndbox, 'xmin')
                        #import pdb;pdb.set_trace()
                        node_xmin.text = str(frame_bbox[0])
                        node_ymin = SubElement(node_bndbox, 'ymin')
                        node_ymin.text = str(frame_bbox[1])
                        node_xmax = SubElement(node_bndbox, 'xmax')
                        node_xmax.text = str(frame_bbox[0]+frame_bbox[2])
                        node_ymax = SubElement(node_bndbox, 'ymax')
                        node_ymax.text = str(frame_bbox[1]+frame_bbox[3])
                        xml = tostring(node_root, pretty_print=True)
                        save_xml=save_xml_path+'/'+video_name+'+'+str(c)+'.xml'
                        with open(save_xml, 'wb') as f:
                            f.write(xml)

                        cv2.waitKey(1)
                        c=c+1
                    else:
                        break
                vc.release()
        print(i,video_name+'finished!')
#制作trainval.txt,train.txt,val.txt,test.txt
def make_txt(path):
    trainval_path=os.path.join(txt_path,'trainval.txt')
    train_path=os.path.join(txt_path,'train.txt')
    val_path=os.path.join(txt_path,'val.txt')
    test_path = os.path.join(txt_path, 'test.txt')
    video_folders = os.listdir(path)
    for i, video_name in enumerate(video_folders):
        video_path = os.path.join(videos_path, video_name)
        files = os.listdir(video_path)
        # 读取对应视频文件的IR_label.json
        # res_file = os.path.join(video_path, 'IR_label.json')
        # with open(res_file, 'r') as f:
        #     label_res = json.load(f)
        for file in files:
            if (file == 'IR.mp4'):
                IR_path = os.path.join(video_path, file)
                vc = cv2.VideoCapture(IR_path)  # 读入IR.mp4视频文件
                c = 0
                rval = vc.isOpened()
                while rval:  # 循环读取视频帧
                    rval, frame = vc.read()
                    if rval:
                        # trainval 80;
                        # train 40;
                        # val 40;
                        # test 20
                        if(i<40):
                            with open(trainval_path,'a') as f_trainval:
                                f_trainval.write(video_name + '+' + str(c))
                                f_trainval.write('\n')
                            with open(train_path,'a') as f_train:
                                f_train.write(video_name + '+' + str(c))
                                f_train.write('\n')
                        elif(i<80):
                            with open(trainval_path,'a') as f_trainval:
                                f_trainval.write(video_name + '+' + str(c))
                                f_trainval.write('\n')
                            with open(val_path,'a') as f_val:
                                f_val.write(video_name + '+' + str(c))
                                f_val.write('\n')
                        else:
                            with open(test_path,'a') as f_test:
                                f_test.write(video_name + '+' + str(c))
                                f_test.write('\n')
                        #cv2.imwrite(pic_path + video_name + '+' + str(c) + '.jpg', frame)
                        cv2.waitKey(1)
                        c = c + 1
                    else:
                        break
                vc.release()
        print(i, video_name + 'finished!')

if __name__=='__main__':
    #import pdb;pdb.set_trace()
    #video_to_img(videos_path)##读取无人机视频文件并将其转换成图片存入data/VOCdevkit/VOC2007/JPEGImages/下
    #save_xml(videos_path)
    make_txt(videos_path)

