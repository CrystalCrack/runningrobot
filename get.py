import xlrd        # 读取excel文件需要的库
from xlutils.copy import copy
import xlwt

#init
i=0
a=1
b=2
c=3
workbook=xlrd.open_workbook('sample.xlsx')



#circle
i+=1

sheets =workbook.sheet_names()  # 获取工作簿中的所有表格
worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
new_workbook=copy(workbook)
new_worksheet=new_workbook.get_sheet(0)
new_worksheet.write(i,0, a )
new_worksheet.write(i,1, b )
new_worksheet.write(i,2, c )


#end getting samples
new_workbook.save('nomovesample.xlsx')#不调整的位置
#new_workbook.save('movesample.xlsx')#调整的位置
















