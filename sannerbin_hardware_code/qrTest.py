import qrcode
# example data
data = "\nhttps://www.\nthepythoncode.\ncom"
# output file name
filename = "site.png"



import qrcode
import json

# JSON object
jsonObj= {
"scan_data": {
"warehouse_name": "",
"sample_id": "PB010701",
"uuid": "b2baacb0-03fc-11ed-960c-e45f011bac6d",
"sample_weight": "0.0",
"commodity_name": "Toor",
"Variety": "White",
"device_serial_no": "agnext_pb01",
"inspection_date": 1657861938000
},
"analysis": [{
"analysisName": "grain_count",
"totalAmount": 31,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "aspect_ratio",
"totalAmount": 1.327,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "length",
"totalAmount": 8.294,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "width",
"totalAmount": 6.573,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "broken",
"totalAmount": 64.5,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "clean",
"totalAmount": 0,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "damaged",
"totalAmount": 0,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "foreign_matters",
"totalAmount": 35.5,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "other",
"totalAmount": 0,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "shrivelled",
"totalAmount": 0,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "weevilled",
"totalAmount": 0,
"analysisType": "VISIO_COUNT"
}, {
"analysisName": "broken",
"totalAmount": "90.8 %",
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "clean",
"totalAmount": "0.0 %",
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "damaged",
"totalAmount": "0.0 %",
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "foreign_matters",
"totalAmount": "9.2 %",
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "other",
"totalAmount": "0.0 %",
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "shrivelled",
"totalAmount": "0.0 %",
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "weevilled",
"totalAmount": "0.0 %",
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "length",
"totalAmount": 8.294,
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "width",
"totalAmount": 6.573,
"analysisType": "VISIO_DENSITY"
}, {
"analysisName": "moisturecontent",
"totalAmount": "0.0",
"analysisType": "MOISTURE_METER"
}]
}

data=json.dumps(jsonObj,indent=1)
print(len(data))
# # generate qr code
# img = qrcode.make(data)
# # save img to a file
# img.save(filename)