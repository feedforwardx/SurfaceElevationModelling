"""
Data parser for ISRO lunar mission data files.
Handles parsing of OATH, OAT, SPM, and LBR files.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


def parse_utc_time_from_parts(parts_list: List[str]) -> datetime:
    """
    Parse UTC time from a list of time components.
    
    Args:
        parts_list: List of time components [year, month, day, hour, minute, second, millisecond]
    
    Returns:
        datetime: Parsed datetime object
    
    Raises:
        ValueError: If parts_list doesn't have exactly 7 elements or parsing fails
    """
    if len(parts_list) != 7:
        raise ValueError("parts_list must have exactly 7 elements")
    
    try:
        year = int(parts_list[0].strip())
        month = int(parts_list[1].strip())
        day = int(parts_list[2].strip())
        hour = int(parts_list[3].strip())
        minute = int(parts_list[4].strip())
        second = int(parts_list[5].strip())
        millisecond = int(parts_list[6].strip())
        return datetime(year, month, day, hour, minute, second, millisecond * 1000)
    except ValueError as e:
        raise ValueError(f"Failed to parse time components: {e}")


def extract_time_parts_from_28char_string(s: str) -> List[str]:
    """
    Extract time parts from a 28-character string.
    
    Args:
        s: 28-character string containing time information
    
    Returns:
        List of time components
    
    Raises:
        ValueError: If string length is not 28 characters
    """
    if len(s) != 28:
        raise ValueError("String must be exactly 28 characters long")
    
    return [s[i:i+4].strip() for i in range(0, 28, 4)]


def parse_oath_header(filepath: str) -> Dict[str, Any]:
    """
    Parse OATH (Orbit Attitude Header) file.
    
    Args:
        filepath: Path to the OATH file
    
    Returns:
        Dictionary containing parsed header information
    """
    header = {}
    
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        
        header['record_type'] = line[0:12].strip()
        header['project_name'] = line[12:33].strip()
        header['block_length_bytes'] = int(line[33:39].strip())
        header['station_id'] = line[39:43].strip()
        header['start_utc'] = parse_utc_time_from_parts(
            extract_time_parts_from_28char_string(line[43:71])
        )
        header['end_utc'] = parse_utc_time_from_parts(
            extract_time_parts_from_28char_string(line[71:99])
        )
        header['num_oat_records'] = int(line[99:105].strip())
        header['record_length_oat'] = int(line[105:111].strip())
        header['attitude_source'] = int(line[111:112].strip())
        header['mission_phase'] = int(line[112:113].strip())
    
    return header


def parse_oat_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse OAT (Orbit Attitude) file.
    
    Args:
        filepath: Path to the OAT file
    
    Returns:
        List of dictionaries containing parsed OAT records
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip().startswith('ORBTATTD'):
                continue
            
            l = line.strip()
            i = 0
            r = {}
            
            r['record_type'] = l[i:i+8].strip(); i += 8
            r['physical_record_num'] = int(l[i:i+6].strip()); i += 6
            r['block_length_bytes'] = int(l[i:i+4].strip()); i += 4
            r['utc_time'] = parse_utc_time_from_parts(
                extract_time_parts_from_28char_string(l[i:i+28])
            ); i += 28
            
            r['lunar_pos_xyz_j2000_earth_kms'] = np.array([
                float(l[i+j:i+j+20].strip()) for j in range(0, 60, 20)
            ]); i += 60
            
            r['satellite_pos_xyz_j2000_kms'] = np.array([
                float(l[i+j:i+j+20].strip()) for j in range(0, 60, 20)
            ]); i += 60
            
            r['satellite_vel_xyz_kms_sec'] = np.array([
                float(l[i+j:i+j+12].strip()) for j in range(0, 36, 12)
            ]); i += 36
            
            r['sc_attitude_q_inertial_to_body'] = np.array([
                float(l[i+j:i+j+14].strip()) for j in range(0, 56, 14)
            ]); i += 56
            
            r['q_earth_fixed_iau'] = np.array([
                float(l[i+j:i+j+14].strip()) for j in range(0, 56, 14)
            ]); i += 56
            
            r['q_lunar_fixed_iau'] = np.array([
                float(l[i+j:i+j+14].strip()) for j in range(0, 56, 14)
            ]); i += 56
            
            r['sub_satellite_lat_deg'] = float(l[i:i+14].strip()); i += 14
            r['sub_satellite_lon_deg'] = float(l[i:i+14].strip()); i += 14
            r['solar_azimuth_deg'] = float(l[i:i+14].strip()); i += 14
            r['solar_elevation_deg'] = float(l[i:i+14].strip()); i += 14
            r['latitude_deg'] = float(l[i:i+14].strip()); i += 14
            r['longitude_deg'] = float(l[i:i+14].strip()); i += 14
            r['satellite_altitude_kms'] = float(l[i:i+12].strip()); i += 12
            i += 52
            
            data.append(r)
    
    return data


def parse_spm_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse SPM (Sun Parameters) file.
    
    Args:
        filepath: Path to the SPM file
    
    Returns:
        List of dictionaries containing parsed SPM records
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip().startswith('ORBTATTD'):
                continue
            
            l = line.strip()
            i = 0
            r = {}
            
            r['record_type'] = l[i:i+8].strip(); i += 8
            r['physical_record_num'] = int(l[i:i+6].strip()); i += 6
            r['block_length_bytes'] = int(l[i:i+4].strip()); i += 4
            r['utc_time'] = parse_utc_time_from_parts(
                extract_time_parts_from_28char_string(l[i:i+28])
            ); i += 28
            
            r['satellite_pos_x_kms'] = float(l[i:i+20].strip()); i += 20
            r['satellite_pos_y_kms'] = float(l[i:i+20].strip()); i += 20
            r['satellite_pos_z_kms'] = float(l[i:i+20].strip()); i += 20
            i += 36
            
            r['phase_angle_deg'] = float(l[i:i+9].strip()); i += 9
            r['sun_aspect_deg'] = float(l[i:i+9].strip()); i += 9
            r['sun_azimuth_deg'] = float(l[i:i+9].strip()); i += 9
            r['sun_elevation_deg'] = float(l[i:i+9].strip()); i += 9
            r['orbit_limb_direction'] = int(l[i:i+1].strip()); i += 1
            
            data.append(r)
    
    return data


def parse_lbr_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse LBR (Lunar Body Reference) file.
    
    Args:
        filepath: Path to the LBR file
    
    Returns:
        List of dictionaries containing parsed LBR records
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip().startswith('ORBTATTD'):
                continue
            
            l = line.strip()
            i = 0
            r = {}
            
            r['record_type'] = l[i:i+8].strip(); i += 8
            r['physical_record_num'] = int(l[i:i+6].strip()); i += 6
            r['block_length_bytes'] = int(l[i:i+4].strip()); i += 4
            r['utc_time'] = parse_utc_time_from_parts(
                extract_time_parts_from_28char_string(l[i:i+28])
            ); i += 28
            
            r['satellite_pos_x_kms'] = float(l[i:i+20].strip()); i += 20
            r['satellite_pos_y_kms'] = float(l[i:i+20].strip()); i += 20
            r['satellite_pos_z_kms'] = float(l[i:i+20].strip()); i += 20
            i += 64
            
            data.append(r)
    
    return data


def find_closest_record(target_time: datetime, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the closest record to a target time.
    
    Args:
        target_time: Target datetime to find closest record for
        records: List of records with 'utc_time' field
    
    Returns:
        Closest record or None if no records found
    """
    closest = None
    min_diff = timedelta.max
    
    for r in records:
        if 'utc_time' in r:
            d = abs(target_time - r['utc_time'])
            if d < min_diff:
                min_diff = d
                closest = r
    
    return closest


class LunarDataParser:
    """
    Main class for parsing ISRO lunar mission data files.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the parser with data directory.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.oath_header = None
        self.oat_data = None
        self.spm_data = None
        self.lbr_data = None
    
    def load_all_data(self) -> bool:
        """
        Load all data files.
        
        Returns:
            True if all files loaded successfully, False otherwise
        """
        try:
            oath_path = f"{self.data_dir}/params.oath"
            oat_path = f"{self.data_dir}/params.oat"
            spm_path = f"{self.data_dir}/sun_params.spm"
            lbr_path = f"{self.data_dir}/params.lbr"
            
            self.oath_header = parse_oath_header(oath_path)
            self.oat_data = parse_oat_file(oat_path)
            self.spm_data = parse_spm_file(spm_path)
            self.lbr_data = parse_lbr_file(lbr_path)
            
            return (self.oath_header is not None and 
                   self.oat_data is not None and 
                   self.spm_data is not None and 
                   self.lbr_data is not None)
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_image_acquisition_data(self) -> Dict[str, Any]:
        """
        Get data relevant to image acquisition time.
        
        Returns:
            Dictionary containing relevant acquisition data
        """
        if not self.oat_data:
            raise ValueError("OAT data not loaded")
        
        image_acquisition_time = self.oat_data[0]['utc_time']
        
        relevant_oat_record = find_closest_record(image_acquisition_time, self.oat_data) if self.oat_data else None
        relevant_spm_record = find_closest_record(image_acquisition_time, self.spm_data) if self.spm_data else None
        relevant_lbr_record = find_closest_record(image_acquisition_time, self.lbr_data) if self.lbr_data else None
        
        return {
            'acquisition_time': image_acquisition_time,
            'oat_record': relevant_oat_record,
            'spm_record': relevant_spm_record,
            'lbr_record': relevant_lbr_record
        } 