import segment_characters
import detect_plate_v2
import pickle
from lib_detection import load_model

test_data = []
test_data.append(("testdata/1.jpg", "30A56999"))
test_data.append(("testdata/2.jpg", "37A02656"))
test_data.append(("testdata/3.jpg", "22L8288"))
test_data.append(("testdata/4.jpg", "95L3456"))
test_data.append(("testdata/5.jpg", "18A00402"))
test_data.append(("testdata/6.jpg", "30F24442"))
test_data.append(("testdata/7.jpg", "MCLRNF1"))
test_data.append(("testdata/8.jpg", "MH01AV8866"))
test_data.append(("testdata/9.jpg", "48A56789"))
test_data.append(("testdata/10.jpg", "15A22222"))
test_data.append(("testdata/11.jpg", "76A07676"))
test_data.append(("testdata/12.jpg", "51F21111"))
test_data.append(("testdata/13.jpg", "51F22160"))
test_data.append(("testdata/14.jpg", "30A77178"))
test_data.append(("testdata/15.jpg", "35A00135"))
test_data.append(("testdata/16.jpg", "51A33888"))
test_data.append(("testdata/17.jpg", "51A12288"))
test_data.append(("testdata/18.jpg", "51F55295"))
test_data.append(("testdata/19.jpg", "89A08888"))
test_data.append(("testdata/20.jpg", "89A09196"))
test_data.append(("testdata/21.jpg", "T1596455"))
test_data.append(("testdata/22.jpg", "37A20700"))
test_data.append(("testdata/23.jpg", "T1576549"))
test_data.append(("testdata/24.jpg", "61T33727"))
test_data.append(("testdata/25.jpg", "43A44444"))
test_data.append(("testdata/26.jpg", "38P105694"))
test_data.append(("testdata/27.jpg", "70C105314"))
test_data.append(("testdata/28.jpg", "50N155555"))
test_data.append(("testdata/29.jpg", "59D277777"))
test_data.append(("testdata/30.jpg", "29U102222"))

def get_plate(model, column_list, characters):
    classification_result = []
    for each_character in characters:
        # converts it to a 1D array
        each_character = each_character.reshape(1, -1);
        result = model.predict(each_character)
        classification_result.append(result)

        plate_string = ''
        for eachPredict in classification_result:
            plate_string += eachPredict[0]

    # print('Predicted license plate')
    # print(plate_string)

    # it's possible the characters are wrongly arranged
    # since that's a possibility, the column_list will be
    # used to sort the letters in the right order

    # print('Length of column_list %d' %len(column_list))
    column_list_copy = column_list[:]
    column_list.sort()
    rightplate_string = ''
    for each in column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]

    # print('License plate')
    # print(rightplate_string)
    return rightplate_string

print("Loading model")
filename = './finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)
print('Model loaded. Predicting characters of number plate')

index = 0
total_passed = 0;

print('Execute test data...')
for test in test_data:
    index = index + 1
    binary = detect_plate_v2.get_plate_area(test[0], wpod_net)
    column_list, characters = segment_characters.get_number_area(binary)
    number = get_plate(model, column_list, characters)
    result = "FAILED"
    if test[1] == number:
        result = "PASSED"
        total_passed = total_passed + 1

    print("Check file %s which is %s - predict is %s: Result: %s at data %d/%d" %(test[0], test[1], number, result, index, len(test_data)))

print("Total result passed: %d/%d" %(total_passed, len(test_data)))


