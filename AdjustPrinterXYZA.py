##########################################################################################        
######################### ddefine the adjustment of X Y Z and Extrusion ##################
##########################################################################################
class AdjustPrinterXYZA:
    def __init__(self,s,incXYZNum,incANum,speed):
        self.s = s
        self.incXYZNum = incXYZNum
        self.incANum   = incANum
        self.speed = speed

############################################ X  ##########################################
    def GoToZeros(self):
        print( 'Set current position as zeros')
        l = 'G90 X0 Y0 Z0' +self.speed
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())           
        return 
    def SetZerosA(self):
        print( 'Set current angle as zeros')
        l = 'G92 A0'
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())        
        return 
    def SetZerosXYZ(self):
        print( 'Set current position as zeros')
        l = 'G92 X0 Y0 Z0 A0'
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())        
        return 
    def X_plus(self,incXYZNum):

        l = 'G91\nG1 X'+incXYZNum+self.speed
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())
        return 
    def X_minus(self,incXYZNum):
        l = 'G91\nG1 X-'+incXYZNum+self.speed
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())
        return 
############################################ Y  #############################################

    def Y_plus(self,incXYZNum):
        l = 'G91\nG1 Y'+incXYZNum+self.speed
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())
        return 
    def Y_minus(self,incXYZNum):
        l = 'G91\nG1 Y-'+incXYZNum+self.speed
        print( 'Sending: ' + l,) 
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())
        return   
############################################ Z  #############################################

    def Z_plus(self,incXYZNum):
        l = 'G91\nG1 Z'+incXYZNum+self.speed
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())
        return 
    def Z_minus(self,incXYZNum):
        l = 'G91\nG1 Z-'+incXYZNum+self.speed
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())
        return 
############################################ Extrude  #############################################
    def A_plus(self,incANum):
        l = 'G91\nG1 A-'+str(float(incANum)/100)+self.speed
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())
        return 
    def A_minus(self,incANum):
        l = 'G91\nG1 A+'+str(float(incANum)/100)+self.speed
        print( 'Sending: ' + l,)
        self.s.write((l + '\n').encode()) # Send g-code block to grbl
        grbl_out = self.s.readline() # Wait for grbl response with carriage return
        print( ' : ' + grbl_out.decode())
        return 
