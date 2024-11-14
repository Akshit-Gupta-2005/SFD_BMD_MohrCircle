import math
import csv
import numpy as np
import pandas as pd


# Function Definitions
def input_beam_length():
    beam_length = float(input("Enter the length of the beam in metres: "))
    return beam_length

def input_point_forces(beam_length):
    number_point_forces = int(input("Enter the number of point forces: "))
    point_force_vector_2d = []
    for i in range(number_point_forces):
        print(f"Enter the magnitude of force {i+1}: ")
        magnitude = float(input())
        print("Enter the distance in metres from the left end of the beam: ")
        distance = float(input())
        if distance <= beam_length:
            point_force_vector_2d.append([magnitude, distance, 0])
            print(f"{magnitude} Newtons {distance} metres from the left")
        else:
            print("The distance of a point force must be less than the length of the beam.")
            i -= 1
    print(f"Number of point forces = {len(point_force_vector_2d)}")
    print("================================================")
    return point_force_vector_2d

def input_udl(beam_length):
    number_udl = int(input("Enter the number of UDL's: "))
    udl_vector_2d = []
    for i in range(number_udl):
        print(f"Enter the magnitude of the UDL {i+1}: ")
        magnitude = float(input())
        print("Enter the starting distance: ")
        start_distance = float(input())
        print("Enter the end distance: ")
        end_distance = float(input())
        if start_distance < beam_length and end_distance <= beam_length:
            if start_distance < end_distance:
                udl_vector_2d.append([magnitude, start_distance, end_distance])
                print(f"{magnitude} Newton Metres {start_distance} Metres to {end_distance} Metres")
            else:
                print("The start distance must be less than the end distance.")
                i =i- 1
        else:
            print("The UDL must be within the beam length.")
            i =i- 1
    print(f"Number of UDL's = {len(udl_vector_2d)}")
    print("================================================")
    return udl_vector_2d

def input_supports(beam_length):
    support_number = 2
    support_distances = []
    support_distances.append(0.0)
    support_distances.append(beam_length)
    print("================================================")
    return support_distances

def udl_to_point(udl_vector_2d, i):
    magnitude = udl_vector_2d[i][0] * (udl_vector_2d[i][2] - udl_vector_2d[i][1])
    position = udl_vector_2d[i][1] + ((udl_vector_2d[i][2] - udl_vector_2d[i][1]) / 2)
    return [magnitude, position, 1]

def compute_sum(point_force_vector_2d):
    return sum(force[0] for force in point_force_vector_2d)

def compute_r_b(point_force_vector_2d, support_distances):
    rb_ccw = sum(
        (support_distances[0] - force[1]) * force[0]
        for force in point_force_vector_2d
        if force[1] < support_distances[0]
    )
    rb_cw = sum(
        (force[1] - support_distances[0]) * force[0]
        for force in point_force_vector_2d
        if force[1] >= support_distances[0]
    )
    return (rb_ccw - rb_cw) / -(support_distances[1] - support_distances[0])

def compute_r_a(r_b, sum_point_forces):
    return sum_point_forces - r_b

def output_reactions(sum_point_forces, r_a, r_b):
    print(f"Sum of Forces = {sum_point_forces}")
    print(f"Ra = {r_a}")
    print(f"Rb = {r_b}")
    print("================================================")

def create_force_vector(point_force_vector_2d, udl_vector_2d, support_distances, r_a, r_b):
    force_vector = []
    for force in point_force_vector_2d:
        if force[2] == 0:
            force_vector.append([force[0], force[1], 0])
    for udl in udl_vector_2d:
        force_vector.append([udl[0], udl[1], udl[2]])
    force_vector.append([-r_a, support_distances[0], 0])
    force_vector.append([-r_b, support_distances[1], 0])
    return sorted(force_vector, key=lambda x: x[1])

def output_force_vector(force_vector):
    for force in force_vector:
        if force[2] == 0:
            print(f"{force[0]} Newtons {force[1]} Metres")
        else:
            print(f"{force[0]} Newtons {force[1]} to {force[2]} Metres")
    print("================================================")

def compute_step_size(beam_length):
    return beam_length / 1000

def compute_vx(force_vector, x, i):
    if force_vector[i][2] == 0:
        return force_vector[i][0]
    elif force_vector[i][2] <= x:
        return force_vector[i][0] * (force_vector[i][2] - force_vector[i][1])
    else:
        return force_vector[i][0] * (x - force_vector[i][1])

def compute_mx(force_vector, x, i):
    if force_vector[i][2] == 0:
        return force_vector[i][0] * (x - force_vector[i][1])
    elif force_vector[i][2] <= x:
        return force_vector[i][0] * (force_vector[i][2] - force_vector[i][1]) * (
            x - (force_vector[i][1] + (force_vector[i][2] - force_vector[i][1]) / 2)
        )
    else:
        return force_vector[i][0] * (x - force_vector[i][1]) * ((x - force_vector[i][1]) / 2)

import os

def output_csv(sfbm_2d):
    if os.path.exists("Data.csv"):
        os.remove("Data.csv")  # Delete old file to ensure data is fresh
    with open("C:\\Users\\akshi\\Documents\\Coding\\VSCode\\MoM project\\Data.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Distance_[m]", "Shear_Force_[N]", "Bending_Moment_[Nm]"])
        writer.writerows(sfbm_2d)


def program_exit():
    input("Enter any character to generate SFD BMD: ")

# Main Function
def main(beam_length):
    point_force_vector_2d = input_point_forces(beam_length)
    udl_vector_2d = input_udl(beam_length)
    support_distances = input_supports(beam_length)

    for i in range(len(udl_vector_2d)):
        point_force_vector_2d.append(udl_to_point(udl_vector_2d, i))

    sum_point_forces = compute_sum(point_force_vector_2d)
    r_b = compute_r_b(point_force_vector_2d, support_distances)
    r_a = compute_r_a(r_b, sum_point_forces)
    output_reactions(sum_point_forces, r_a, r_b)

    force_vector = create_force_vector(point_force_vector_2d, udl_vector_2d, support_distances, r_a, r_b)
    output_force_vector(force_vector)

    step_size = compute_step_size(beam_length)
    sfbm_2d = []
    vx_max, mx_max = 0, 0

    for x in range(0, int(beam_length / step_size) + 1):
        x *= step_size
        vx, mx = 0, 0
        for i in range(len(force_vector)):
            if x >= force_vector[i][1]:
                vx += compute_vx(force_vector, x, i)
                mx += compute_mx(force_vector, x, i)
            else:
                break
        sfbm_2d.append([x, vx, mx])
        vx_max = max(vx_max, abs(vx))
        mx_max = max(mx_max, abs(mx))

    print(f"V(x)_max (N) = {vx_max}")
    print(f"M(x)_max (Nm) = {mx_max}")
    print("================================================")

    output_csv(sfbm_2d)
    
def axial_force_distribution(A_f, A_x, beam_length, support_ans):
    step_size = compute_step_size(beam_length)
    axial_force_vector = []

    for x in np.arange(0, beam_length + step_size, step_size):
        if support_ans == 1:  # Simply supported
            if 0 <= x <= A_x:
                axial_force = A_f
            else:
                axial_force = 0
        elif support_ans == 2:  # Fixed-end beam
            if 0 <= x <= A_x:
                axial_force = A_f * (beam_length - A_x) / beam_length
            elif A_x < x <= beam_length:
                axial_force = -A_f * A_x / beam_length
            else:
                axial_force = 0
        else:
            raise ValueError("Invalid support type. Use 1 for simply supported or 2 for fixed.")

        axial_force_vector.append([x, axial_force])

    return axial_force_vector
beam_length = float(input("Enter the length of the beam in metres: "))
if __name__ == "__main__":
    main(beam_length)
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\\Users\\akshi\\Documents\\Coding\\VSCode\\MoM project\\Data.csv')

# Extracting data for plotting
distance = data['Distance_[m]']
shear_force = data['Shear_Force_[N]']
bending_moment = data['Bending_Moment_[Nm]']

# Plotting Shear Force Diagram (SFD)
plt.figure(figsize=(12, 6))

# Shear Force Diagram
plt.subplot(1, 2, 1)
plt.plot(distance, shear_force, marker='o', linestyle='-')
plt.title('Shear Force Diagram (SFD)')
plt.xlabel('Distance [m]')
plt.ylabel('Shear Force [N]')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()

# Bending Moment Diagram (BMD)
plt.subplot(1, 2, 2)
plt.plot(distance, bending_moment, marker='o', linestyle='-')
plt.title('Bending Moment Diagram (BMD)')
plt.xlabel('Distance [m]')
plt.ylabel('Bending Moment [Nm]')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()

plt.tight_layout()
plt.show()



A_f = float(input("Enter the magnitude of the axial force (in N): "))
A_x = float(input("Enter the distance of application from the left end (in m): "))
support_ans = int(input("Enter the type of support (1 for simply supported, 2 for fixed): "))
axial_forces = axial_force_distribution(A_f, A_x, beam_length, support_ans)
def append_axial_force_to_csv(axial_force_vector, csv_path):
    # Load the existing data from the CSV file
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
    else:
        print(f"Error: {csv_path} does not exist.")
        return
    
    # Create a DataFrame for the axial force data
    axial_force_df = pd.DataFrame(axial_force_vector, columns=['Distance_[m]', 'Axial_Force_[N]'])

    # Merge the data on 'Distance_[m]'
    merged_data = pd.merge(existing_data, axial_force_df, on='Distance_[m]', how='left')

    # Fill NaN values in case of mismatch (optional)
    merged_data['Axial_Force_[N]'].fillna(0, inplace=True)

    # Save the updated data back to the CSV file
    merged_data.to_csv(csv_path, index=False)
    print("Axial force data appended to the existing CSV file.")
    
    return axial_force_df

axial_force_df=append_axial_force_to_csv(axial_forces,  r'C:\\Users\\akshi\\Documents\\Coding\\VSCode\\MoM project\\Data.csv')

# Plot axial force distribution (optional)
plt.figure(figsize=(10, 5))
plt.plot(axial_force_df['Distance_[m]'], axial_force_df['Axial_Force_[N]'], marker='o', linestyle='-')
plt.title('Axial Force Distribution')
plt.xlabel('Distance [m]')
plt.ylabel('Axial Force [N]')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid()
plt.show()

h=float(input("Enter the Height of cross section of the beam : "))
b=float(input("Enter the Base length of cross section of the beam : "))

Mohr_x=float(input("Enter the Point on the beam where you want to generate Mohr Circle (in accuracy of 0.01) : "))
newdata=pd.read_csv(r'C:\\Users\\akshi\\Documents\\Coding\\VSCode\\MoM project\\Data.csv')
values=newdata.loc[newdata['Distance_[m]']==Mohr_x]
print(values)
#values['Shear_Force_[N]'].values[0]

import numpy as np
import matplotlib.pyplot as plt

import math


import math
import numpy as np
import matplotlib.pyplot as plt

def principal_stresses(nx, ny, tau):
    p1 = (nx + ny) / 2 + math.sqrt(((nx - ny) / 2) ** 2 + tau ** 2)
    p2 = (nx + ny) / 2 - math.sqrt(((nx - ny) / 2) ** 2 + tau ** 2)
    p_avg = (p1 + p2) / 2
    tau_max = math.sqrt(((nx - ny) / 2) ** 2 + tau ** 2)
    
    # Calculate theta_p with division by zero check
    if (nx - ny) / 2 == 0:
        theta_p = math.pi / 2  # 90 degrees in radians
    else:
        theta_p = math.atan(tau / ((nx - ny) / 2)) / 2
    
    # Calculate theta_s with division by zero check
    if tau == 0:
        theta_s = math.pi / 2  # 90 degrees in radians
    else:
        theta_s = math.atan((-(nx - ny) / 2) / tau) / 2

    return p1, p2, p_avg, tau_max, theta_p, theta_s

def MohrsCirclePlaneStress(nx, ny, tau):
    # Calculate average normal
    sigma_avg = (nx + ny) / 2
    # Calculate Radius
    R = np.sqrt(((nx - ny) / 2) ** 2 + tau ** 2)
    # Call principal stress function to get principal stresses
    p1, p2, p_avg, tau_max, theta_p, theta_s = principal_stresses(nx, ny, tau)
    # Equation of circle
    y1 = np.linspace(-2.5 * abs(tau_max), 2.5 * abs(tau_max), num=1000)
    y2 = np.linspace(2.5 * abs(tau_max), -2.5 * abs(tau_max), num=1000)
    sigma_xprime1 = np.ones(y1.shape) * sigma_avg + np.sqrt(R ** 2 - y1 ** 2)
    sigma_xprime2 = np.ones(y2.shape) * sigma_avg - np.sqrt(R ** 2 - y2 ** 2)
    # Plot Circle
    plt.plot(sigma_xprime1, y1, label='shear(normal)')
    plt.plot(sigma_xprime2, y2, label='shear(normal)')
    plt.grid()
    plt.title("Mohr's Circle")
    plt.xlabel('sigma (normal stress)')
    plt.ylabel('-tau (shear stress)')
    plt.ylim([-2.5 * abs(tau_max), 2.5 * abs(tau_max)])
    # Plot important points
    plt.plot(p1, 0, '.', markersize=20)
    plt.plot(p2, 0, '.', markersize=20)
    plt.plot(p_avg, 0, '.', markersize=20)
    plt.plot(p_avg, -tau_max, '.', markersize=20)
    plt.plot(p_avg, tau_max, '.', markersize=20)
    plt.plot(nx, tau, '.', markersize=20)
    plt.plot(ny, -tau, '.', markersize=20)
    plt.legend(['shear(normal)', 'shear(normal)', 'principal stress 1', 'principal stress 2',
                'average normal stress', '(p avg, -tau max)', '(p avg, tau max)',
                'REF1(nx, tau)', 'REF2(ny, tau)'])
    plt.axis('equal')
    plt.show()


y=float(input("Enter the distance from neutral axis where you want to generate Mohr Circle : "))

axial_F=float((values['Axial_Force_[N]'].values[0]))
Bending_M=float((values['Bending_Moment_[Nm]'].values[0]))
Shear_F=float((values['Shear_Force_[N]'].values[0]))

nx = (axial_F/(h*b)) - (12*Bending_M*y)/(b*h*h*h)

ny = 0

tau = (6*Shear_F*(((h/2)*(h/2))-(y*y)))/(b*h*h*h)

print(f"sigmax={nx},sigmay={ny},shear stress= {tau}")

MohrsCirclePlaneStress(nx,ny,tau)