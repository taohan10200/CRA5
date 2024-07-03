proxy=dict(type='normal',    #'normal' 'special', 'direct'
                normal='http://10.1.11.100:8086/',
                special='', #special='http://10.3.3.20:25800/'
            )

storage=dict(
    type='local', #'s3' 'local'
    s3=None,
    local = '../data/ERA5'
)

normalization = False
isobaricInhPa = None


vnames=dict(
    pressure=['z','q', 'u', 'v', 't', 'r','w'],
    single=['v10','u10','v100', 'u100', 't2m','tcc', 'sp','tp', 'msl']) # 'tisr'


# total_levels = [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
#  775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
#  350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
#  70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
#  1.]


pressure_request_dic={'product_type': 'reanalysis',
              'variable': [
                            # 'divergence',
                           # 'fraction_of_cloud_cover',
                           'geopotential',
                           # 'ozone_mass_mixing_ratio',
                           # 'potential_vorticity',
                           'relative_humidity',
                           # 'specific_cloud_ice_water_content',
                           # 'specific_cloud_liquid_water_content',
                           'specific_humidity',
                           # 'specific_rain_water_content',
                           # 'specific_snow_water_content',
                           'temperature',
                           'u_component_of_wind',
                           'v_component_of_wind',
                           'vertical_velocity',
                           # 'vorticity',
                           ],
              'pressure_level': ['1',
                                 '2',
                                 '3',
                                 '5',
                                 '7',
                                 '10',
                                 '20',
                                 '30',
                                 '50',
                                 '70',
                                 '100',
                                 '125',
                                 '150',
                                 '175',
                                 '200',
                                 '225',
                                 '250',
                                 '300',
                                 '350',
                                 '400',
                                 '450',
                                 '500',
                                 '550',
                                 '600',
                                 '650',
                                 '700',
                                 '750',
                                 '775',
                                 '800',
                                 '825',
                                 '850',
                                 '875',
                                 '900',
                                 '925',
                                 '950',
                                 '975',
                                 '1000',
                                 ],
              'time': ['00:00','01:00','02:00','03:00','04:00','05:00',
                       '06:00','07:00','08:00','09:00','10:00','11:00',
                       '12:00','13:00','14:00','15:00','16:00','17:00',
                       '18:00','19:00','20:00','21:00','22:00','23:00',
                       ],

              'month' : [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
               ],
              'format': 'netcdf',
              }

['mean_wave_direction', 'mean_wave_period', 'significant_height_of_combined_wind_waves_and_swell']

single_request_dic = {'product_type':'reanalysis',
              'variable': [
                  '2m_temperature',
                  '10m_u_component_of_wind',
                  '10m_v_component_of_wind',
                  '100m_u_component_of_wind',
                  '100m_v_component_of_wind',
                  'total_cloud_cover',
                  'surface_pressure',
                  'mean_sea_level_pressure',
                  'total_precipitation',
                  ],     # 'toa_incident_solar_radiation'
                         
              'time':
                  [
                  '00:00', '01:00', '02:00',
                  '03:00', '04:00', '05:00',
                  '06:00', '07:00', '08:00',
                  '09:00', '10:00', '11:00',
                  '12:00', '13:00', '14:00',
                  '15:00', '16:00', '17:00',
                  '18:00', '19:00', '20:00',
                  '21:00', '22:00', '23:00',
              ],

              'month' : [
                  '01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
              ],
              'day': [
                  '01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
                  '13', '14', '15',
                  '16', '17', '18',
                  '19', '20', '21',
                  '22', '23', '24',
                  '25', '26', '27',
                  '28', '29', '30',
                  '31',
              ],
              'format': 'netcdf', #netcdf
              }