from PIL import Image
import numpy as np
import bitstring
import cv2


# Contains cl
class ETLn_Record:
    def read(self, bs, pos=None):
        if pos:
            bs.bytepos = pos * self.octets_per_record

        r = bs.readlist(self.bitstring)

        record = dict(zip(self.fields, r))

        self.record = {
            k: (self.converter[k](v) if k in self.converter else v)
            for k, v in record.items()
        }
        return self.record

    def get_image(self):
        return self.record['Image Data']

    def get_class(self):
        return self.record['JIS Code']


class ETL167_Record(ETLn_Record):
    def __init__(self):
        self.octets_per_record = 2052
        self.fields = [
            "Data Number", "Character Code", "Serial Sheet Number", "JIS Code", "EBCDIC Code",
            "Evaluation of Individual Character Image", "Evaluation of Character Group",
            "Male-Female Code", "Age of Writer", "Serial Data Number",
            "Industry Classification Code", "Occupation Classification Code",
            "Sheet Gatherring Date", "Scanning Date",
            "Sample Position Y on Sheet", "Sample Position X on Sheet",
            "Minimum Scanned Level", "Maximum Scanned Level", "Image Data"
        ]
        self.bitstring = 'uint:16,bytes:2,uint:16,hex:8,hex:8,4*uint:8,uint:32,4*uint:16,4*uint:8,pad:32,bytes:2016,pad:32'
        self.converter = {
            'Character Code': lambda x: x.decode('ascii'),
            'Image Data': lambda x: Image.eval(Image.frombytes('F', (64, 63), x, 'bit', 4).convert('L'),
                                               lambda x: x * 16)
        }

    def get_char(self):
        return bytes.fromhex(self.record['JIS Code']).decode('iso2022_jp')


# Instantiates the matrix with placeholder values using 0s
# Parameters:
# Total records = 71959
# X dim = 64
# Y dim = 63
# Classes in EBCDIC Code
training_data = np.zeros([71959, 48, 48])

# Total records = 71959
training_labels = np.zeros([71959, ], dtype='uint8')

# Loops through ETL1C_7 to ETL1C_13 files and grabs their information into the matrix above
label = -1
rec_num = 0
previous_class = ""
current_index = 0
for i in range(7, 14):
    file = 'ETL-1/ETL1C_{Number:02d}'.format(Number=i)
    bit_stream = bitstring.ConstBitStream(filename=file)

    etln_record = ETL167_Record()

    if i == 13:
        rec_num = 4233
    else:
        rec_num = 11288

    for j in range(rec_num):
        try:
            # Grabs raw data from the record in the current index i
            raw_data = etln_record.read(bit_stream, j)

            print(raw_data)

            # Gets the image data in that record
            raw_image = np.array(etln_record.get_image().getdata()).reshape(63, 64).astype("float32")
            image_data = cv2.resize(raw_image, dsize=(48, 48))

            # Loops through the image_data and puts them into the training data
            y_size, x_size = image_data.shape
            for y in range(y_size):
                for x in range(x_size):
                    training_data[current_index, y, x] = image_data[y, x]

            # Gets the image type in EBCDIC form
            current_class = etln_record.get_class()

            # Assigns the EBCDIC code in the class
            if current_class == 'b4' and i != 7:
                training_labels[current_index, ] = 3
            elif current_class == 'b2' and i != 7:
                training_labels[current_index, ] = 1
            elif current_class == 'b3' and i != 7:
                training_labels[current_index, ] = 2
            else:
                # Detects a change in character at the start of the ETL file and increments
                if j == 0:
                    label += 1
                elif j != 0:
                    # Detects a change in character within the ETL file and increments label
                    previous_obj = etln_record.read(bit_stream, j - 1)
                    previous_class = etln_record.get_class()

                    if previous_class != current_class:
                        label += 1

                training_labels[current_index, ] = label

            current_index += 1
        except bitstring.ReadError:
            # Should only have 2 empty records: 33863 and 67727
            print("Empty record found in: " + str(previous_class))
            pass

# Saves into a NPZ file called training_data.npz
print("Saving data to training_data.npz...")
np.savez('training_data.npz', training_data)
np.savez('training_labels.npz', training_labels)
print("Saving complete")
