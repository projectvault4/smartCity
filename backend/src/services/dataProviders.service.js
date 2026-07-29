const waqiService = require('./waqi.service');
const openWeatherService = require('./openWeather.service');
const googleTrafficService = require('./googleTraffic.service');

module.exports = {
  fetchAqi: waqiService.fetchAqi,
  fetchCurrentWeather: openWeatherService.fetchCurrentWeather,
  fetchTrafficRoute: googleTrafficService.fetchTrafficRoute
};
