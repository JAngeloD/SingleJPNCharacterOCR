from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import bitstring


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


####################################################################################################################
# TEMPORARY CODE FOR DISPLAYING DATA
####################################################################################################################
file_temp = 'ETL-1/ETL1C_{Number:02d}'.format(Number=7)
f_temp = bitstring.ConstBitStream(filename=file_temp)

etln_record = ETL167_Record()

# Image data
read = etln_record.read(f_temp, 9287)
image = etln_record.get_image()

image_data = image.getdata()
image_matrix = np.array(image_data)

print(read)

####################################################################################################################
# TEMPORARY CODE FOR DISPLAYING DATA
####################################################################################################################
file_temp = 'ETL-1/ETL1C_{Number:02d}'.format(Number=7)
f_temp = bitstring.ConstBitStream(filename=file_temp)

etln_record = ETL167_Record()

# Image data
read = etln_record.read(f_temp, 9287)
image = etln_record.get_image()

image_data = image.getdata()
image_matrix = np.array(image_data)

print(read)

plt.imshow(image_matrix.reshape(63, 64), cmap='gray')
plt.show()
