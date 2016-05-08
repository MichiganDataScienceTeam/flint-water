from xml.dom.minidom   import parseString

def CheckDir(fdir):
   import os
   try: 
      if not os.path.exists(fdir): os.makedirs(fdir)
   except OSError: pass

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def readString(tag_name,file_name = 'InputParameters.xml'):
    file      = open(file_name,'r')
    data_file = file.read()
    file.close()
    data = parseString(data_file)
    xmlTag = data.getElementsByTagName(tag_name)[0]
    tag_name_value = getText(xmlTag.childNodes)
    return tag_name_value

def readFloat(tag_name,file_name = 'InputParameters.xml'):
    file      = open(file_name,'r')
    data_file = file.read()
    file.close()
    data = parseString(data_file)
    xmlTag = data.getElementsByTagName(tag_name)[0]
    tag_name_value = float(getText(xmlTag.childNodes))
    return tag_name_value

def readInt(tag_name,file_name = 'InputParameters.xml'):
    file      = open(file_name,'r')
    data_file = file.read()
    file.close()
    data = parseString(data_file)
    xmlTag = data.getElementsByTagName(tag_name)[0]
    tag_name_value = int(getText(xmlTag.childNodes))
    return tag_name_value

def readBool(tag_name,file_name = 'InputParameters.xml'):
    file      = open(file_name,'r')
    data_file = file.read()
    file.close()
    data = parseString(data_file)
    xmlTag = data.getElementsByTagName(tag_name)[0]
    tag_name_value = getText(xmlTag.childNodes)
    if (tag_name_value == "F" or tag_name_value == "f" or tag_name_value == "False" or tag_name_value == "false" or tag_name_value == "0"):
       return False
    else: 
        return True

def Read_Integer_Input(string_value):
    while True:
      try:
         int_value = int(raw_input(string_value))
         break
      except (TypeError, ValueError):
         print ("Not an integer. Try Again.")
    return int_value

def Read_Float_Input(string_value):
    while True:
      try:
         float_value = float(raw_input(string_value))
         break
      except (TypeError, ValueError):
         print ("Not a real number. Try Again.")
    return float_value

def Read_String_Input(string_value):
    while True:
      try:
         str_value = raw_input(string_value)
         break
      except (TypeError, ValueError):
         print ("Not a string. Try Again.")
    return str_value

def Read_YN_Input(string_value):
    while True:
      try:
         X = raw_input(string_value)
         if((X == 'y') or (X == 'Y')):
           bool_value = True
           break
         elif((X == 'n') or (X == 'N')):
           bool_value = False
           break
         else:
           print ("Invalid Input. Try Again. (Please enter Y, y, N or n)")
      except (TypeError, ValueError):
         print ("Invalid Input. Try Again. (Please enter Y, y, N or n)")
    return bool_value
