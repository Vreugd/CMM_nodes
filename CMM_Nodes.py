# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 07:50:59 2023

@author: jan.devreugd@tno.nl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import pyperclip
import io

def funcAsphere(x,y,RoC,conic,offset):
    R = np.sqrt(x**2+y**2)
    asphereZ = R**2 / ( RoC * ( 1 + np.sqrt( 1 - (1+conic) * (R/RoC)**2  ) ) ) + offset
    return asphereZ  
    
def PlotContour(x,y,z,title):

    fig, ax = plt.subplots(figsize=(10, 8))
    p1 = ax.tripcolor(x,y,z,cmap=plt.cm.jet, shading='gouraud')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(title)
    ax.set_aspect('equal', 'box')
    fig.colorbar(p1)
    plt.scatter(x, y, color='black', s=2)
    plt.tight_layout()
    st.pyplot(plt)  
    
def main():
    
    with st.sidebar:
        st.title('CMM nodes for facesheet measurements')
        st.write('info: jan.devreugd@tno.nl')
        
        RoC = st.number_input('Radius of Curvature [meters]:',value = -4.19818, format='%.5f', step = 0.001)
        conic = st.number_input('conical constant [-]:',value = -3.604, format='%.4f', step = 0.001)
        Rout = st.number_input('Outer radius mirror [meters]:',value = 0.302, format='%.3f', step = 0.001)
        n_rings = st.number_input('number of rings [-]:',value = 40, step=1,max_value=70)
        offset = st.number_input('z-offset [meters]:', value = 0.0, format='%.5f', step = 0.01)
        
        pitch = Rout/n_rings
        
        st.write(f'pitch = {1000*pitch:2f} [mm]')
        
        r = 0
        x = np.array([])
        y = np.array([])
        for j in range(1,n_rings+1):
            r = r + pitch
            nact = 6*j
            theta_offset = 0.0
            for k in range(1,nact+1):
                phi = np.pi*2/(nact)*(k + theta_offset)
                x = np.append(x,r*np.cos(phi))
                y = np.append(y,r*np.sin(phi))
                
        R = np.sqrt(x**2 + y**2)
        Z = funcAsphere(x, y, RoC, conic, offset)
        
        factor = -1
        #factors = 1
        
        Z1X = factor*(2*x/(RoC*(np.sqrt(1 - (conic + 1)*(x**2 + y**2)/RoC**2) + 1)) + x*(conic + 1)*(x**2 + y**2)/(RoC**3*np.sqrt(1 - (conic + 1)*(x**2 + y**2)/RoC**2)*(np.sqrt(1 - (conic + 1)*(x**2 + y**2)/RoC**2) + 1)**2))
        Z1Y = factor*(2*y/(RoC*(np.sqrt(1 - (conic + 1)*(x**2 + y**2)/RoC**2) + 1)) + y*(conic + 1)*(x**2 + y**2)/(RoC**3*np.sqrt(1 - (conic + 1)*(x**2 + y**2)/RoC**2)*(np.sqrt(1 - (conic + 1)*(x**2 + y**2)/RoC**2) + 1)**2))
        Z1Z = factor*(-1*np.ones((len(x))))
        VL = np.sqrt(Z1X**2+Z1Y**2+Z1Z**2)
        
        Z1X = Z1X/VL
        Z1Y = Z1Y/VL
        Z1Z = Z1Z/VL
        
        my_array = np.array([1000*x,1000*y,1000*Z,Z1X,Z1Y,Z1Z]).T
        
        st.write(f'{len(x)} nodes')
        
    #plotly_function(x, y, 'CMM nodes')
    PlotContour(x, y, Z, f'{len(x)} datapoints, pitch = {1000*pitch:.2f} [mm]\n RoC = {RoC} [m], kappa = {conic}, max Radius = {Rout} [m], z-offset = {offset} [m]')


    # Display the array using Streamlit with the header
    with st.expander('Data Points', expanded=False): 
        df = pd.DataFrame(my_array, columns=['x [mm]', 'y [mm]', 'z [mm]','e1','e2','e3'])
        st.table(df)
    
    tsv_content = io.StringIO()
    np.savetxt(tsv_content, my_array, delimiter='\t', fmt='%.8f')  # Use "\t" for tab-separated values
    tsv_content.seek(0)
    
    bytes_data = io.BytesIO(tsv_content.getvalue().encode())
    st.download_button(
           label="Download Data in .txt File Format",
           data=bytes_data,
           file_name=f'CMM_input_data_{len(x)}_nodes.txt',
           mime="text/csv"
       )

if __name__ == "__main__":
    main()
