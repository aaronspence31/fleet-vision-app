// Wrapper component for the signin page

"use client";
import React, { useState } from "react";
import { Box, Typography, Divider } from "@mui/material";
import Image from "next/image";
import { DataGrid, GridToolbarQuickFilter } from "@mui/x-data-grid";
import styles from "./page.module.css";
import { vehicleColumns, vehicleData, Vehicle, VehicleGridInfo } from "./types";
import {
  SpeedOutlined,
  PersonOutline,
  PaletteOutlined,
  RectangleOutlined,
  HandymanOutlined,
  AbcOutlined,
} from "@mui/icons-material";
import VehicleHealthList from "@/components/Dashboard/VehicleHealthList/VehicleHealthList";
import {
  VehicleDisplayError,
  vehicleErrorData,
} from "@/components/Dashboard/VehicleHealthList/types";

const QuickSearchToolbar = () => {
  return (
    <Box sx={{}} className={styles.searchBar}>
      <GridToolbarQuickFilter
        style={{ flex: 1 }}
        quickFilterParser={(searchInput: string) =>
          searchInput
            .split(",")
            .map((value) => value.trim())
            .filter((value) => value !== "")
        }
        debounceMs={400}
      />
    </Box>
  );
};

const noVehicleSelected = () => {
  return (
    <Box>
      <Typography fontSize={32} textAlign="center">
        No Vehicle Selected
      </Typography>
      <Image
        src="/images/carIcon.png"
        alt="Description of the image"
        width={600}
        height={400}
        style={{ opacity: 0.5 }}
      />
      <Typography textAlign="center">
        Select a vehicle to see a detailed view
      </Typography>
    </Box>
  );
};

const vehicleInfoGrid = (selectedVehicle: Vehicle) => {
  const gridData: VehicleGridInfo[][] = [
    [
      { value: selectedVehicle.odometer + "km", icon: <SpeedOutlined /> },
      { value: selectedVehicle.currentDriver, icon: <PersonOutline /> },
    ],
    [
      { value: selectedVehicle.licensePlate, icon: <RectangleOutlined /> },
      { value: selectedVehicle.color, icon: <PaletteOutlined /> },
    ],
    [
      { value: selectedVehicle.make, icon: <HandymanOutlined /> },
      { value: selectedVehicle.model, icon: <AbcOutlined fontSize="large" /> },
    ],
  ];
  return (
    <Box>
      <Divider flexItem />
      {gridData.map((gridRow: VehicleGridInfo[]) => (
        <>
          <Box className={styles.infoContainer}>
            <Box className={styles.innerInfoContainer}>
              {gridRow[0].icon}
              <Typography noWrap>{gridRow[0].value}</Typography>
            </Box>
            <Divider orientation="vertical" flexItem />
            <Box className={styles.innerInfoContainer}>
              {gridRow[1].icon}
              <Typography>{gridRow[1].value}</Typography>
            </Box>
          </Box>
          <Divider flexItem />
        </>
      ))}
    </Box>
  );
};

export default function Vehicles() {
  const [selectedVehicle, setSelectedVehicle] = useState<Vehicle | null>(null);

  const generateErrorArr = (): VehicleDisplayError[] => {
    const shuffled = [...vehicleErrorData].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, 4);
  };

  return (
    <Box className={styles.container}>
      <Box
        className={styles.leftBox}
        sx={{
          "& .header": {
            backgroundColor: "var(--accent-primary-hover)",
            color: "white",
          },
        }}
      >
        <DataGrid
          disableColumnFilter
          disableColumnSelector
          disableDensitySelector
          disableMultipleRowSelection
          disableColumnResize
          disableColumnMenu
          hideFooter
          autosizeOnMount
          columns={vehicleColumns}
          rows={vehicleData}
          slots={{ toolbar: QuickSearchToolbar }}
          onRowSelectionModelChange={(id) => {
            const selectedRowData = vehicleData.filter(
              (row) => id[0] == row.id
            )[0];
            setSelectedVehicle(selectedRowData);
          }}
          slotProps={{
            toolbar: {
              showquickfilter: true,
            },
          }}
          sx={{
            "& .MuiDataGrid-columnHeader:focus, .MuiDataGrid-cell:focus": {
              outline: "none",
            },
            "& .Mui-selected": {
              backgroundColor: "var(--accent-primary) !important",
              color: "white",
            },
            "& .MuiDataGrid-row:hover": {
              backgroundColor: "rgba(61, 139, 255, 0.5)",
              cursor: "pointer",
            },
          }}
        />
      </Box>
      <Box className={styles.rightBox}>
        {selectedVehicle ? (
          <Box
            display={"flex"}
            justifyContent={"space-between"}
            alignItems={"center"}
            flexDirection={"column"}
            height="100%"
          >
            <div>
              <Typography fontSize={32} textAlign="center">
                {selectedVehicle?.name}
              </Typography>
              <Box className={styles.status}>
                <Typography className={styles.statusTitle}>
                  {selectedVehicle.status}
                </Typography>
                <Box
                  className={styles.statusIcon}
                  bgcolor={selectedVehicle.status == "Online" ? "green" : "red"}
                />
              </Box>
              {vehicleInfoGrid(selectedVehicle)}
            </div>
            <div>
              <Box className={styles.vehicleMaintenance}>
                <VehicleHealthList
                  vehicleErrors={generateErrorArr()}
                  vehicleSelected={true}
                />
              </Box>
            </div>
          </Box>
        ) : (
          noVehicleSelected()
        )}
      </Box>
    </Box>
  );
}
