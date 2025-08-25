const nameInput = document.getElementById("emp-name");
const ageInput = document.getElementById("emp-age");
const cityInput = document.getElementById("emp-city");
const addEmp = document.getElementById("add-employee");
const table = document.getElementById("employee-table");
const tbody = table.querySelector('tbody')

addEmp.addEventListener("click", function(){
    const name = nameInput.value.trim();
    const age = ageInput.value.trim();
    const city = cityInput.value.trim();

    if(name && age && city){
        const row = document.createElement('tr');
        row.innerHTML=<td>${name}</td><td>${age}</td><td>${city}</td>;
        tbody.appendChild(row);
        table.style.display='';
        nameInput.value = '';
        ageInput.value= '';
        cityInput.value=''
    }

});