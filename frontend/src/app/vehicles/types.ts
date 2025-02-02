export interface Vehicle {
    id: string
    name: string
    licensePlate: string
    status: 'Online' | 'Offline'
    odometer: number
    make: string
    model: string
    color: string
    currentDriver: string
    driverID: string
    year: number
}

interface VehicleColumns {
    field: string
    headerName: string
    flex: number
    headerClassName: string
}

export interface VehicleGridInfo{
    value: string | number
    icon: React.JSX.Element

}

export const vehicleColumns: VehicleColumns[] = [
    {
      field: "name",
      headerName: "Name",
      flex: 1,
      headerClassName: 'header'
    },
    {
      field: "licensePlate",
      headerName: "License Plate",
      flex: 1,
      headerClassName: 'header'
    },
    {
        field: "currentDriver",
        headerName: "Current Driver",
        flex: 1,
        headerClassName: 'header'
      },
    {
      field: "status",
      headerName: "Status",
      flex: 1,
      headerClassName: 'header'
    },
  ]

export const vehicleData: Vehicle[] = [
    {
      id: 'a7d1e890-2b6a-11eb-adc1-0242ac120002',
      name: 'Toyota Corolla 1',
      licensePlate: 'ABC123',
      status: 'Online',
      odometer: 12000,
      make: 'Toyota',
      model: 'Corolla',
      color: 'Blue',
      currentDriver: 'John Doe',
      driverID: 'b3b61f40-2b6a-11eb-adc1-0242ac120002',
      year: 2020
    },
    {
      id: 'c8d2e990-2b6a-11eb-adc1-0242ac120002',
      name: 'Honda Civic 1',
      licensePlate: 'DEF456',
      status: 'Offline',
      odometer: 15000,
      make: 'Honda',
      model: 'Civic',
      color: 'Red',
      currentDriver: 'Jane Smith',
      driverID: 'd7e3f0a0-2b6a-11eb-adc1-0242ac120002',
      year: 2019
    },
    {
      id: 'e8d4f0b0-2b6a-11eb-adc1-0242ac120002',
      name: 'Ford Focus 1',
      licensePlate: 'GHI789',
      status: 'Online',
      odometer: 18000,
      make: 'Ford',
      model: 'Focus',
      color: 'White',
      currentDriver: 'Jim Brown',
      driverID: 'f9e5g1b0-2b6a-11eb-adc1-0242ac120002',
      year: 2018
    },
    {
      id: '0a3f1c20-2b6b-11eb-adc1-0242ac120002',
      name: 'Chevrolet Malibu 1',
      licensePlate: 'JKL012',
      status: 'Offline',
      odometer: 21000,
      make: 'Chevrolet',
      model: 'Malibu',
      color: 'Black',
      currentDriver: 'Sara White',
      driverID: '1b4g2d30-2b6b-11eb-adc1-0242ac120002',
      year: 2021
    },
    {
      id: '2c5h3e40-2b6b-11eb-adc1-0242ac120002',
      name: 'Nissan Altima 1',
      licensePlate: 'MNO345',
      status: 'Online',
      odometer: 23000,
      make: 'Nissan',
      model: 'Altima',
      color: 'Silver',
      currentDriver: 'David Green',
      driverID: '3d6i4f50-2b6b-11eb-adc1-0242ac120002',
      year: 2020
    },
    {
      id: '4e7j5g60-2b6b-11eb-adc1-0242ac120002',
      name: 'Mazda 3 1',
      licensePlate: 'PQR678',
      status: 'Offline',
      odometer: 25000,
      make: 'Mazda',
      model: '3',
      color: 'Gray',
      currentDriver: 'Chris Black',
      driverID: '5f8k6h70-2b6b-11eb-adc1-0242ac120002',
      year: 2022
    },
    {
      id: '6g9l7i80-2b6b-11eb-adc1-0242ac120002',
      name: 'Hyundai Sonata 1',
      licensePlate: 'STU901',
      status: 'Online',
      odometer: 27000,
      make: 'Hyundai',
      model: 'Sonata',
      color: 'Green',
      currentDriver: 'Emma Blue',
      driverID: '7h0m8j90-2b6b-11eb-adc1-0242ac120002',
      year: 2017
    },
    {
      id: '8i1n9k00-2b6b-11eb-adc1-0242ac120002',
      name: 'Kia Optima 1',
      licensePlate: 'VWX234',
      status: 'Offline',
      odometer: 29000,
      make: 'Kia',
      model: 'Optima',
      color: 'Yellow',
      currentDriver: 'Mike Grey',
      driverID: '9j2o0l10-2b6b-11eb-adc1-0242ac120002',
      year: 2016
    },
    {
      id: '0k3p1m20-2b6c-11eb-adc1-0242ac120002',
      name: 'Subaru Impreza 1',
      licensePlate: 'YZA567',
      status: 'Online',
      odometer: 31000,
      make: 'Subaru',
      model: 'Impreza',
      color: 'Blue',
      currentDriver: 'Nancy Orange',
      driverID: '1l4q2n30-2b6c-11eb-adc1-0242ac120002',
      year: 2015
    },
    {
      id: '2m5r3o40-2b6c-11eb-adc1-0242ac120002',
      name: 'Volkswagen Jetta 1',
      licensePlate: 'BCD890',
      status: 'Offline',
      odometer: 33000,
      make: 'Volkswagen',
      model: 'Jetta',
      color: 'Red',
      currentDriver: 'Olivia Purple',
      driverID: '3n6s4p50-2b6c-11eb-adc1-0242ac120002',
      year: 2014
    },
    {
      id: '4o7t5q60-2b6c-11eb-adc1-0242ac120002',
      name: 'BMW 3 Series 1',
      licensePlate: 'EFG123',
      status: 'Online',
      odometer: 35000,
      make: 'BMW',
      model: '3 Series',
      color: 'White',
      currentDriver: 'Paul Pink',
      driverID: '5p8u6r70-2b6c-11eb-adc1-0242ac120002',
      year: 2013
    },
    {
      id: '6q9v7s80-2b6c-11eb-adc1-0242ac120002',
      name: 'Audi A4 1',
      licensePlate: 'HIJ456',
      status: 'Offline',
      odometer: 37000,
      make: 'Audi',
      model: 'A4',
      color: 'Black',
      currentDriver: 'Sam Red',
      driverID: '7r0w8t90-2b6c-11eb-adc1-0242ac120002',
      year: 2012
    },
    {
      id: '8s1x9u00-2b6c-11eb-adc1-0242ac120002',
      name: 'Mercedes-Benz C-Class 1',
      licensePlate: 'KLM789',
      status: 'Online',
      odometer: 39000,
      make: 'Mercedes-Benz',
      model: 'C-Class',
      color: 'Silver',
      currentDriver: 'Lucas Brown',
      driverID: '9t2y0v10-2b6c-11eb-adc1-0242ac120002',
      year: 2011
    },
    {
      id: '0u3z1w20-2b6d-11eb-adc1-0242ac120002',
      name: 'Lexus IS 1',
      licensePlate: 'NOP012',
      status: 'Offline',
      odometer: 41000,
      make: 'Lexus',
      model: 'IS',
      color: 'Gray',
      currentDriver: 'Liam Yellow',
      driverID: '1v4a2x30-2b6d-11eb-adc1-0242ac120002',
      year: 2010
    },
    {
      id: '2w5b3y40-2b6d-11eb-adc1-0242ac120002',
      name: 'Tesla Model 3 1',
      licensePlate: 'QRS345',
      status: 'Online',
      odometer: 43000,
      make: 'Tesla',
      model: 'Model 3',
      color: 'Green',
      currentDriver: 'Sophia Blue',
      driverID: '3x6c4z50-2b6d-11eb-adc1-0242ac120002',
      year: 2019
    },
    {
      id: '4y7d5a60-2b6d-11eb-adc1-0242ac120002',
      name: 'Jeep Wrangler 1',
      licensePlate: 'TUV678',
      status: 'Offline',
      odometer: 45000,
      make: 'Jeep',
      model: 'Wrangler',
      color: 'Yellow',
      currentDriver: 'Jackson Pink',
      driverID: '5z8e6b70-2b6d-11eb-adc1-0242ac120002',
      year: 2021
    },
    {
      id: '6a9f7c80-2b6d-11eb-adc1-0242ac120002',
      name: 'Toyota Corolla 2',
      licensePlate: 'WXY901',
      status: 'Online',
      odometer: 47000,
      make: 'Toyota',
      model: 'Corolla',
      color: 'Blue',
      currentDriver: 'Daniel Grey',
      driverID: '7b0g8d90-2b6d-11eb-adc1-0242ac120002',
      year: 2018
    },
    {
      id: '8c1h9e00-2b6d-11eb-adc1-0242ac120002',
      name: 'Honda Civic 2',
      licensePlate: 'ZAB234',
      status: 'Offline',
      odometer: 49000,
      make: 'Honda',
      model: 'Civic',
      color: 'Red',
      currentDriver: 'Ethan Brown',
      driverID: '9d2i0f10-2b6d-11eb-adc1-0242ac120002',
      year: 2017
    },
    {
      id: '0e3j1g20-2b6e-11eb-adc1-0242ac120002',
      name: 'Ford Focus 2',
      licensePlate: 'BCD567',
      status: 'Online',
      odometer: 51000,
      make: 'Ford',
      model: 'Focus',
      color: 'White',
      currentDriver: 'Amelia White',
      driverID: '1f4k2h30-2b6e-11eb-adc1-0242ac120002',
      year: 2016
    },
    {
      id: '2g5l3i40-2b6e-11eb-adc1-0242ac120002',
      name: 'Chevrolet Malibu 2',
      licensePlate: 'EFG890',
      status: 'Offline',
      odometer: 53000,
      make: 'Chevrolet',
      model: 'Malibu',
      color: 'Black',
      currentDriver: 'Mason Blue',
      driverID: '3h6m4j50-2b6e-11eb-adc1-0242ac120002',
      year: 2015
    },
    {
      id: '4i7n5k60-2b6e-11eb-adc1-0242ac120002',
      name: 'Nissan Altima 2',
      licensePlate: 'HIJ123',
      status: 'Online',
      odometer: 55000,
      make: 'Nissan',
      model: 'Altima',
      color: 'Silver',
      currentDriver: 'Harper Black',
      driverID: '5j8o6l70-2b6e-11eb-adc1-0242ac120002',
      year: 2014
    }
  ];
 
  
