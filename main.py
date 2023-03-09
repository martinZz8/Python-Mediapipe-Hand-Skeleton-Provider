import cv2
import mediapipe as mp
import io
from os import listdir, mkdir
from os.path import isfile, isdir, join as joinpath

# -- Constants definitions --
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# 21 fingers enums (each has '.name' and '.value' property)
list_of_fingers_enums = [
    mp_hands.HandLandmark.WRIST,
    mp_hands.HandLandmark.THUMB_CMC,
    mp_hands.HandLandmark.THUMB_MCP,
    mp_hands.HandLandmark.THUMB_IP,
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_MCP,
    mp_hands.HandLandmark.INDEX_FINGER_PIP,
    mp_hands.HandLandmark.INDEX_FINGER_DIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_MCP,
    mp_hands.HandLandmark.RING_FINGER_PIP,
    mp_hands.HandLandmark.RING_FINGER_DIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_MCP,
    mp_hands.HandLandmark.PINKY_PIP,
    mp_hands.HandLandmark.PINKY_DIP,
    mp_hands.HandLandmark.PINKY_TIP
]

# -- Options --
is_save_skeleton_data = True  # True
is_save_image_with_skeleton = False
is_draw_skeleton = False
max_num_hands = 1

# -- Defining folder names -- (these folder have to be created!)
IMAGES_FN = "data/images/"
SKELETONS_FN = "data/skeletons/"

# Get inside folder names
folder_names = [f for f in listdir(IMAGES_FN) if isdir(joinpath(IMAGES_FN, f))]

# -- Starting proper script for points determination --
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5
) as hands:
    for fn_idx, folder_name in enumerate(folder_names):
        print(f"{fn_idx+1} of {len(folder_names)}")
        print(f"Processing folder: {folder_name} ...")
        # Determine files inside specific dir
        inner_folder_name = IMAGES_FN + folder_name
        file_names = [f for f in listdir(inner_folder_name) if isfile(joinpath(inner_folder_name, f))]
        file_names = list(filter(lambda x: ".png" in x, file_names))

        # Process each file individually
        # NOTE: Start processing from this 'for', when you don't want to search in multiple directories
        #       remove creation of directories
        #       also remove 'folder_name' concatenation in determining output file names: 'output_txt_file_path' and 'output_image_file_path'
        for idx, file_name in enumerate(file_names):
            print(f"Processing file: {file_name} ...")
            # Get file_name without extension
            splitted_file_name = file_name.split(".")
            file_name_wo_ext = ".".join(splitted_file_name[0:(len(splitted_file_name)-1)])

            # Read an image, flip it around y-axis for correct handedness output (see above).
            input_file_path = IMAGES_FN + "/" + folder_name + "/" + file_name
            image = cv2.flip(cv2.imread(input_file_path), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness, determine shape of image and copy the image
            #print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            # Save points for each hand
            for idx2, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine hand type ("Left", "Right")
                hand_type = results.multi_handedness[idx2].classification[0].label

                # Create 21x2 list
                hand_skeleton_points = [[0] * 2 for i in range(21)]

                # Fill hand_skeleton_points list
                for idx3, finger_enum in enumerate(list_of_fingers_enums):
                    hand_skeleton_points[idx3][0] = round(hand_landmarks.landmark[finger_enum].x * image_width, 2)
                    hand_skeleton_points[idx3][1] = round(hand_landmarks.landmark[finger_enum].y * image_height, 2)
                # print(f"hand_skeleton_points: {hand_skeleton_points}")

                # Draw landmarks
                if is_save_image_with_skeleton:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Write hand skeleton data to .txt file
                if is_save_skeleton_data:
                    # Create output folder for skeleton
                    try:
                        folder_to_create_path = SKELETONS_FN + folder_name
                        mkdir(folder_to_create_path)
                    except FileExistsError as err:
                        print(f"Warning: Directory {folder_to_create_path} already exists!")

                    # Form path for file
                    output_txt_file_path = SKELETONS_FN + folder_name + "/" + file_name_wo_ext
                    if idx2 > 0:
                        output_txt_file_path += str(idx2)
                    output_txt_file_path += ".txt"

                    # Create file and save data to it
                    f = io.open(output_txt_file_path, mode="w", encoding="utf-8")
                    for idx3, finger_point in enumerate(hand_skeleton_points):
                        for idx4, finger_point_coord in enumerate(finger_point):
                            val_to_save = str(finger_point_coord)
                            if idx4 != (len(finger_point) - 1):
                                val_to_save += " "
                            f.write(val_to_save)
                        if idx3 != (len(hand_skeleton_points) - 1):
                            f.write("\n")
                    f.close()

            # Write image with landmarks
            if is_save_image_with_skeleton:
                output_image_file_path = SKELETONS_FN + folder_name + "/" + file_name
                cv2.imwrite(output_image_file_path, cv2.flip(annotated_image, 1))

            # Draw hand world landmarks.
            if results.multi_hand_world_landmarks and is_draw_skeleton:
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
        if idx == (len(file_names) - 1):
            print("----------")
print("End of script!")
