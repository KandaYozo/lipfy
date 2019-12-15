import cv2
import numpy as np
import skimage.io as io


#Skin segmintation function following the lead of "Human Skin Detecion using RGB, HSV and YCbCr color models S. Kolkur 1 , D. Kalbande 2 , P. Shimpi 2 , C. Bapat 2 , and J. Jatakia 2"

def segment(img):
    B=img[:,:,0]
    G=img[:,:,1]
    R=img[:,:,2]

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]

    Y=0.299*R+0.587*G+0.114*B
    Cb=(B-Y)*0.564+128
    Cr=(R-Y)*0.713+128


# # 0.0 <= H <= 50.0 and 0.23 <= S <= 0.68 and
# # R > 95 and G > 40 and B > 20 and R > G and R > B
# # and | R - G | > 15 and A > 15

    skin_one = np.zeros((len(img),len(img[0])))
    skin_one_h = np.logical_and(h>=0.0,h<=50.0)
    skin_one_s = np.logical_and(s>=0.23,s<=0.68)
    skin_one_rgb =  np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(R>95,G>40),B>20),R>G),R>B),np.absolute(np.array(R)-np.array(G)) > 15)
    skin_one = np.logical_and(np.logical_and(skin_one_h,skin_one_rgb),skin_one_s)

# # OR

# # R > 95 and G > 40 and B > 20 and R > G and R > B
# # and | R - G | > 15 and A > 15 and Cr > 135 and
# # Cb > 85 and Y > 80 and Cr <= (1.5862*Cb)+20 and
# # Cr>=(0.3448*Cb)+76.2069 and
# # Cr >= (-4.5652*Cb)+234.5652 and
# # Cr <= (-1.15*Cb)+301.75 and
# # Cr <= (-2.2857*Cb)+432.85

    skin_two = np.zeros((len(img),len(img[0])))
    skin_two_rgb = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(R>95,G>40),B>20),R>G),R>B),np.absolute(np.array(R)-np.array(G))>15)
    skin_two_YCbCr = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(Cr>135,Cb>85),Y>80),Cr<=((1.5862*Cb)+20)),Cr>=((0.3448*Cb)+76.2069)),
                                    Cr>=((-4.5652*Cb)+234.5652)),Cr<=((-1.15*Cb)+301.75)),Cr<=((-2.2857*Cb)+432.85))
    skin_two = np.logical_and(skin_two_YCbCr,skin_two_rgb)
    skin = np.logical_or(skin_one,skin_two)
    skin_image = np.copy(img)
    skin_image[skin] = 255
    skin_image[np.logical_not(skin)] = 0
    return skin_image





def fill_holes(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    res = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    return res


def get_box(img,RGB_img):
    bounding_boxes = []
    masks = np.zeros(img.shape).astype("uint8")
    output = cv2.connectedComponentsWithStats(img, 8 , cv2.CV_32S)
    index = 0
    num_labels = output[0]
    labels = output[1]
    stats = output[2]

    if num_labels>0:
        box_ratio_upper_bound = 1.1
        box_ratio_lower_bound = 0.4
        area=0
        index=0
        for i in range(0, labels.max()+1):
            # if stats[i,cv2.CC_STAT_AREA]>area:
            #     area=stats[i,cv2.CC_STAT_AREA]
            #     index=i
        # labels[labels!=index]=0
        # label_hue = np.uint8(179 * labels / np.max(labels))
        # blank_ch = 255 * np.ones_like(label_hue)
        # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # labeled_img[label_hue == 0] = 0
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area=stats[i,cv2.CC_STAT_AREA]
            if x+w < y+h:
                minor_axis = x+w
                major_axis = y+h
            else:
                minor_axis = y+h
                major_axis = x+w
            if((w/h < box_ratio_upper_bound)and (w/h > box_ratio_lower_bound) and (minor_axis/major_axis > 0.25) and (minor_axis/major_axis<0.97) and area > 300):
                if w/h <= 0.7:
                    h = int(h- 0.3*h)
                # print("----------")
                # print(minor_axis/major_axis)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                mask = np.zeros(img.shape).astype("uint8")
                # print(x,x+w)
                # print("************")
                # print(y,y+h)
                mask[y:y+h,x:x+w] = 255
                bounding_boxes.append([y,y+h,x,x+w])
                # cv2.imshow("mask"+str(index),mask)
                if  not((mask == 255).all()):
                    masks+= mask
                index+=1
                # masks[mask ==255] = 255
                # bounding_boxes.append([x,y,x+w,y+y])
        masks//=255
        masks[masks==1] = 255

        # masks[masks>1] = 0
        # masks[masks<1] = 255
        masks[masks>0] = 255
        # cv2.imshow("rect",img)
        img[masks==0] = 0
        cv2.imshow("RGB",img)
        return bounding_boxes
        # img[np.logical_and(masks==255,img==255)] = 255
        # img[np.logical_not(np.logical_and(masks==255,img==255))] =0







def draw_RGB_with_Rect(RGB_image,Boundary_boxes):
    temp_RGB = np.copy(RGB_image)
    for box in Boundary_boxes:
        cv2.rectangle(RGB_image, (box[2], box[0]), (box[3], box[1]), (255, 255, 255), 2) 


def crop_face(img,boundig_boxes,RGB_image):
    faces = []
    colored_faces = []
    index_of_faces = 0
    masks = np.zeros(img.shape)
    for box in boundig_boxes:
        faces.append(img[box[0]:box[1],box[2]:box[3]])
        masks [box[0]:box[1],box[2]:box[3]] = 255
        colored_faces.append(RGB_image[box[0]:box[1],box[2]:box[3]])
        cv2.imshow("face"+str(index_of_faces) , faces[index_of_faces])
        index_of_faces+=1
    img = cv2.GaussianBlur(img,(3,3),0)
    RGB_image[masks == 0] = 0
    cv2.imshow("final",RGB_image)
    for face in faces:
        temp = np.copy(face)
        face[temp==255] = 0
        face[temp==0] = 255
        output = cv2.connectedComponentsWithStats(face, 3 , cv2.CV_32S)
        index = 0
        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        smallest_label = 1+ np.argmin(stats[1:cv2.CC_STAT_AREA])
        face[labels == largest_label] = 0
        # cv2.imshow("faces with eyes"+str(index_of_faces) , face)
        index_of_faces+=1

    index_of_faces = 0
    for face in faces:
        output = cv2.connectedComponentsWithStats(face, 8 , cv2.CV_32S)
        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        if num_labels>0:
            for i in range(1, labels.max()+1):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cv2.rectangle(face, (x, y), (x + w, y + h), (255, 255, 255), 2)
        # cv2.imshow("faces with eyes"+str(index_of_faces) , face)
        index_of_faces+=1

    # for colored_face in colored_faces:
    #     B=colored_face[:,:,0]
    #     G=colored_face[:,:,1]
    #     R=colored_face[:,:,2]

    #     Y=0.299*R+0.587*G+0.114*B
    #     Cb=(B-Y)*0.564+128
    #     Cr=(R-Y)*0.713+128
    #     average_luminance = luminance.sum()/(colored_face.shape[0]*colored_face.shape[1])
    #     luminance = np.array(Y)
    #     taw = 1
    #     if average_luminance < 64:
    #         taw = 1.5
    #     elif average_luminance > 190:
    #         taw = 0.7
    #     R_dash = np.power(R,taw)
    #     G_dash = np.power(G,taw)
    #     nominator = 0.5*(2*R_dash-G_dash-B)
    #     (R_dash-B)*(G_dash-B)
    #     doneminator = np.sqrt((np.power((R_dash-G_dash),2) + (R_dash-B)*(G_dash-B)))
    #     Inverse_theta = np.divide(nominator,doneminator)
    #     theta = np.arccos(Inverse_theta)
    #     colored_face[theta < 90 ] = 0
    #     cv2.imshow("mouthes"+str(index_of_faces) , colored_face)
    #     index_of_faces+=1
    
    grads = []
    grades_index = 0
    for colored_face in colored_faces:
        colored_face = cv2.cvtColor(colored_face,cv2.COLOR_RGB2GRAY)
        colored_face = cv2.GaussianBlur(colored_face,(3,3),0)
        grad_x = cv2.Sobel(colored_face,cv2.CV_16S,1,0,ksize=3)
        grad_y = cv2.Sobel(colored_face,cv2.CV_16S,0,1,ksize=3)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        
        grads.append(cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0))
        v_x = np.sum(grads[grades_index],axis=0)
        h_y = np.sum(grads[grades_index],axis=1)
        max_x = np.amax(v_x)
        range_of_x = v_x[v_x >= (max_x/3)]
        smallest_x = np.amin(range_of_x)
        smallest_x_boundary = np.where(v_x==smallest_x)[0]
        largest_x = np.amax(range_of_x)
        largest_x_boundary = np.where(v_x==largest_x)[0]
        range_of_y = h_y[h_y>=(0.05*(largest_x_boundary-smallest_x_boundary))]
        smallest_y = np.amin(range_of_y)
        samllest_y_boundary = np.where(h_y==smallest_y)[0]
        largest_y_boundary = np.argmax(grads[grades_index])
        cv2.rectangle(RGB_image,(smallest_x_boundary,samllest_y_boundary),(largest_x_boundary,largest_y_boundary),(255,255,255),2)  
        # cv2.imshow("face"+str(grades_index) , RGB_image)
        grades_index+=1  

        # zeros_intersection = np.where(v_x==0)
        # # print(v_x)
        # # print(zeros_intersection)
        # difference = np.diff(zeros_intersection)
        # # intersection = np.where(difference > 1)
        # intersection = difference[difference > 1]
    # print(difference)
    # print(intersection)
    # regions_in_x = 
    # max_x = np.amax(v_x)
    # range_of_x = v_x[v_x >= (max_x/3)]
    # smallest_x = np.amin(range_of_x)
    # smallest_x_boundary = np.where(v_x==smallest_x)[0]
    # largest_x = np.amax(range_of_x)
    # largest_x_boundary = np.where(v_x==largest_x)[0]
    # print(np.where(v_x==smallest_x),np.where(v_x==largest_x))
    # range_of_y = h_y[h_y>=(0.05*(largest_x_boundary-smallest_x_boundary))]
    # smallest_y = np.amin(range_of_y)
    # samllest_y_boundary = np.where(h_y==smallest_y)[0]
    # largest_y_boundary = np.argmax(grad)
    # print(np.where(h_y==smallest_y),largest_y_boundary)
    # cv2.rectangle(grad,(smallest_x_boundary,samllest_y_boundary),(largest_x_boundary,largest_y_boundary),(255,255,255),2))

    # print(v_x)
    # print("************")
    # print(h_y)
def put_dog_filter(dog,fc,x,y,w,h):
    face_width = w
    face_height = h
    
    dog = cv2.resize(dog,(int(face_width*1.5),int(face_height*1.75)))
    for i in range(int(face_height*1.75)):
        for j in range(int(face_width*1.5)):
            for k in range(3):
                if dog[i][j][k]<235:
                    fc[y+i-int(0.375*h)-1][x+j-int(0.25*w)][k] = dog[i][j][k]
    return fc
def put_hat(hat,fc,x,y,w,h):
    
    face_width = w
    face_height = h
    
    hat_width = face_width+1
    hat_height = int(0.35*face_height)+1
    
    hat = cv2.resize(hat,(hat_width,hat_height))
    
    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat[i][j][k]<235:
                    fc[y+i-int(0.25*face_height)][x+j][k] = hat[i][j][k]
    return fc
def put_moustache(mst,fc,x,y,w,h):
    
    face_width = w
    face_height = h


    mst_width = int(face_width*0.4166666)+1
    mst_height = int(face_height*0.142857)+1


    mst = cv2.resize(mst,(mst_width,mst_height))

    for i in range(int(0.62857142857*face_height),int(0.62857142857*face_height)+mst_height):
        for j in range(int(0.29166666666*face_width),int(0.29166666666*face_width)+mst_width):
            for k in range(3):
                if mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k] <235:
                    fc[y+i][x+j][k] = mst[i-int(0.62857142857*face_height)][j-int(0.29166666666*face_width)][k]
    return fc

mst = io.imread("C:/Users/Kanda/Desktop/proj/Image Project (Abdelgawad)/moustache.png")
dog = io.imread("C:/Users/Kanda/Desktop/proj/Image Project (Abdelgawad)/dog_filter.png")
hat = io.imread("C:/Users/Kanda/Desktop/proj/Image Project (Abdelgawad)/cowboy_hat.png")
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    
    RGB_image = img
    skin_image = segment(RGB_image)
    holes_filled_skin_image = fill_holes(skin_image)
    holes_filled_skin_image = holes_filled_skin_image.astype(np.uint8)
    holes_filled_skin_image = cv2.cvtColor(holes_filled_skin_image,cv2.COLOR_RGB2GRAY)
    boundig_boxes = get_box(holes_filled_skin_image,RGB_image)
    draw_RGB_with_Rect(RGB_image,boundig_boxes)
    x = 0
    xw = 0
    y = 0
    yh = 0
    w = 0
    h = 0
    for box in boundig_boxes:
        y = box[0]
        yh = box[1]
        x = box[2]
        xw = box[3]
        h = np.abs(yh-y)
        w = np.abs(xw-x)
    #RGB_image = put_moustache(mst, RGB_image,x,y,w,h)
    #RGB_image = put_hat(hat, RGB_image, x, y, w, h)
    #RGB_image = put_dog_filter(dog, RGB_image, x, y, w, h)
    cv2.imshow("final",RGB_image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


# RGB_image = cv2.imread("3-people.jpg")
# skin_image = segment(RGB_image)
# holes_filled_skin_image = fill_holes(skin_image)
# holes_filled_skin_image = holes_filled_skin_image.astype(np.uint8)
# holes_filled_skin_image = cv2.cvtColor(holes_filled_skin_image,cv2.COLOR_RGB2GRAY)
# boundig_boxes = get_box(holes_filled_skin_image,RGB_image)
# draw_RGB_with_Rect(RGB_image,boundig_boxes)
# cv2.imshow("final",RGB_image)
# # crop_face(holes_filled_skin_image,boundig_boxes,RGB_image)


# cv2.waitKey(0)