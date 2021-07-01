import Vue from 'vue'
import App from './App.vue'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'

Vue.config.productionTip = false


import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
Vue.use(ElementUI)

/* leaflet icon */
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png')
})


new Vue({
  render: h => h(App),
}).$mount('#app')
